"""Train one matched Level-1 coordinate-aware Dense-Scout variant."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ism_diffusion.scale_data import (
    load_parent_split,
    sample_windows,
    variant_geometry,
)
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.scale_model import (
    CoordinateDenseDenoiser,
    CoordinateDenoiserConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


def learning_rate(config: dict, step: int, total_steps: int) -> float:
    peak = float(config["learning_rate"])
    floor = float(config.get("minimum_learning_rate", 0.0))
    warmup = int(config.get("warmup_steps", 0))
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    model_state = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.lerp_(model_state[name].detach(), 1.0 - decay)
    model_buffers = dict(model.named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(model_buffers[name])


def state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(
    model,
    ema,
    optimizer,
    config,
    step,
    history,
    best_validation,
    initialization_hash,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "history": history,
        "best_validation": best_validation,
        "initialization_hash": initialization_hash,
    }


def build_validation_batches(
    parents: np.ndarray,
    variant: str,
    definitions: list[dict],
    seed: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batches = {}
    for index, definition in enumerate(definitions):
        coordinate_stride = int(definition["coordinate_stride"])
        if definition.get("unit_for_punit", False) and variant == "Punit":
            coordinate_stride = 1
        batches[str(definition["name"])] = sample_windows(
            parents,
            width=int(definition["width"]),
            spin_stride=int(definition["spin_stride"]),
            coordinate_stride=coordinate_stride,
            batch_size=int(definition["samples"]),
            rng=np.random.default_rng(seed + 1009 * index),
            device=device,
            augment=False,
        )
    return batches


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = str(config["variant"])
    if variant not in {"T0", "T3", "Pphase", "Punit"}:
        raise ValueError("variant must be T0, T3, Pphase, or Punit")
    seed = int(config.get("seed", 1234))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    requested_device = args.device or config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    parent_path = resolve_path(args.config, config["parent_data"])
    output_dir = resolve_path(args.config, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_parents = load_parent_split(parent_path, "train")
    val_parents = load_parent_split(parent_path, "val")
    if train_parents.lattice_size != val_parents.lattice_size:
        raise ValueError("train and validation parent sizes differ")

    model_config = CoordinateDenoiserConfig(**config["model"])
    model = CoordinateDenseDenoiser(model_config).to(device)
    initialization_hash = state_fingerprint(model)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = CoordinateAbsorbingDiffusion(
        t_min=float(config.get("t_min", 0.01)),
        t_max=float(config.get("t_max", 1.0)),
        full_mask_probability=float(config.get("full_mask_probability", 0.02)),
    )
    optimizer_options = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(config.get("adam_betas", [0.9, 0.95])),
        "weight_decay": float(config.get("weight_decay", 0.05)),
    }
    fused = bool(config.get("fused_adamw", True)) and device.type == "cuda"
    if fused:
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    total_steps = int(args.steps or config["training_steps"])
    widths = [int(value) for value in config["widths"]]
    strides = [int(value) for value in config["strides"]]
    batch_sizes = {int(key): int(value) for key, value in config["batch_sizes"].items()}
    accumulation = int(config.get("gradient_accumulation", 2))
    grad_clip = float(config.get("gradient_clip", 1.0))
    ema_decay = float(config.get("ema_decay", 0.999))
    validation_interval = int(config.get("validation_interval", 500))
    checkpoint_interval = int(config.get("checkpoint_interval", 2000))
    precision = str(config.get("precision", "bfloat16"))
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if device.type == "cuda" and precision == "bfloat16"
        else nullcontext
    )
    training_rng = np.random.default_rng(seed + 1)
    device_generator = torch.Generator(device=device).manual_seed(seed + 2)
    validation_batches = build_validation_batches(
        val_parents.spins,
        variant,
        list(config["validation_geometries"]),
        seed + 10_000,
        device,
    )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    header = {
        "variant": variant,
        "device": str(device),
        "parameters": parameters,
        "initialization_hash": initialization_hash,
        "parent_data": str(parent_path),
        "train_parents": len(train_parents.spins),
        "val_parents": len(val_parents.spins),
        "steps": total_steps,
    }
    print(json.dumps(header))
    (output_dir / "run_config.json").write_text(
        json.dumps({**config, "effective_steps": total_steps}, indent=2),
        encoding="utf-8",
    )
    history: list[dict] = []
    best_validation = float("inf")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for step in range(1, total_steps + 1):
        model.train()
        width, spin_stride, coordinate_stride = variant_geometry(
            variant, widths, strides, training_rng
        )
        batch_size = batch_sizes[width]
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_ce = 0.0
        for _ in range(accumulation):
            clean, coordinates, valid_mask = sample_windows(
                train_parents.spins,
                width=width,
                spin_stride=spin_stride,
                coordinate_stride=coordinate_stride,
                batch_size=batch_size,
                rng=training_rng,
                device=device,
                augment=True,
            )
            with autocast():
                result = diffusion.training_loss(
                    model,
                    clean,
                    coordinates,
                    valid_mask,
                    generator=device_generator,
                )
                scaled_loss = result.loss / accumulation
            scaled_loss.backward()
            accumulated_loss += float(result.loss.detach().item()) / accumulation
            accumulated_ce += float(result.masked_ce.detach().item()) / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = learning_rate(config, step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        update_ema(ema, model, ema_decay)

        if step == 1 or step % validation_interval == 0 or step == total_steps:
            ema.eval()
            validation: dict[str, dict] = {}
            for name, (clean, coordinates, valid_mask) in validation_batches.items():
                score, per_t = diffusion.validation_nelbo(
                    ema,
                    clean,
                    coordinates,
                    valid_mask,
                    t_grid=config["validation_t_grid"],
                    seed=seed + 20_000,
                    batch_size=int(config.get("validation_batch_size", 4)),
                )
                validation[name] = {
                    "nelbo": score,
                    "per_t": {str(key): value for key, value in per_t.items()},
                }
            anchor_validation = float(validation["anchor_w48_s1"]["nelbo"])
            record = {
                "step": step,
                "variant": variant,
                "width": width,
                "spin_stride": spin_stride,
                "coordinate_stride": coordinate_stride,
                "train_loss": accumulated_loss,
                "masked_ce": accumulated_ce,
                "gradient_norm": float(grad_norm),
                "learning_rate": lr,
                "elapsed_seconds": time.perf_counter() - started,
                "validation": validation,
            }
            history.append(record)
            print(json.dumps(record))
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if anchor_validation < best_validation:
                best_validation = anchor_validation
                torch.save(
                    checkpoint_payload(
                        model,
                        ema,
                        optimizer,
                        config,
                        step,
                        history,
                        best_validation,
                        initialization_hash,
                    ),
                    output_dir / "best_val.pt",
                )
        if step % checkpoint_interval == 0 or step == total_steps:
            torch.save(
                checkpoint_payload(
                    model,
                    ema,
                    optimizer,
                    config,
                    step,
                    history,
                    best_validation,
                    initialization_hash,
                ),
                output_dir / "last.pt",
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    performance = {
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    }
    (output_dir / "performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )
    print(json.dumps({"performance": performance}))


if __name__ == "__main__":
    main()
