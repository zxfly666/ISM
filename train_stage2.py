"""Train a resumable Stage-2 local--global scale denoiser."""

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

from ism_diffusion.scale_data import load_parent_split, sample_windows
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.stage2_data import (
    STAGE2_VARIANTS,
    sample_random_gap_windows,
    sample_random_pe_control_windows,
    sample_stage2_batch,
)
from ism_diffusion.stage2_model import (
    LocalGlobalDenoiserConfig,
    LocalGlobalScaleDenoiser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
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
    model_parameters = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.lerp_(model_parameters[name].detach(), 1.0 - decay)
    model_buffers = dict(model.named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(model_buffers[name])


def state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def rng_payload(
    training_rng: np.random.Generator,
    device_generator: torch.Generator,
) -> dict:
    return {
        "python": random.getstate(),
        "numpy": training_rng.bit_generator.state,
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "device_generator": device_generator.get_state(),
    }


def restore_rng(
    state: dict,
    training_rng: np.random.Generator,
    device_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    training_rng.bit_generator.state = state["numpy"]
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    device_generator.set_state(state["device_generator"])


def checkpoint_payload(
    model,
    ema,
    optimizer,
    config,
    step,
    history,
    best_validation,
    initialization_hash,
    training_rng,
    device_generator,
    elapsed_seconds,
) -> dict:
    return {
        "model_type": "local_global",
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": int(step),
        "history": history,
        "best_validation": float(best_validation),
        "initialization_hash": initialization_hash,
        "rng": rng_payload(training_rng, device_generator),
        "elapsed_seconds": float(elapsed_seconds),
    }


def variant_coordinate_mode(variant: str) -> str:
    if variant.endswith(("T3", "Matched")):
        return "matched"
    if variant.endswith("RandPE"):
        return "randomized"
    return "unit"


def build_validation_batches(
    parents: np.ndarray,
    variant: str,
    definitions: list[dict],
    seed: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output = {}
    coordinate_mode = variant_coordinate_mode(variant)
    for index, definition in enumerate(definitions):
        rng = np.random.default_rng(seed + 1009 * index)
        geometry = str(definition.get("geometry", "uniform"))
        width = int(definition["width"])
        samples = int(definition["samples"])
        spacing = int(definition.get("spacing", definition.get("spin_stride", 1)))
        definition_mode = str(definition.get("coordinate_mode", "variant"))
        mode = coordinate_mode if definition_mode == "variant" else definition_mode
        if geometry == "uniform":
            coordinate_stride = spacing if mode == "matched" else 1
            batch = sample_windows(
                parents,
                width=width,
                spin_stride=spacing,
                coordinate_stride=coordinate_stride,
                batch_size=samples,
                rng=rng,
                device=device,
                augment=False,
            )
        elif geometry == "random_gap":
            batch = sample_random_gap_windows(
                parents,
                width=width,
                gaps=list(definition.get("gaps", [spacing])),
                fixed_gap=definition.get("fixed_gap"),
                coordinate_mode=mode,
                batch_size=samples,
                rng=rng,
                device=device,
                augment=False,
            )
        elif geometry == "random_pe":
            batch = sample_random_pe_control_windows(
                parents,
                width=width,
                gaps=list(definition.get("gaps", [spacing])),
                batch_size=samples,
                rng=rng,
                device=device,
                augment=False,
            )
        else:
            raise ValueError(f"unknown validation geometry: {geometry}")
        output[str(definition["name"])] = batch
    return output


@torch.no_grad()
def gate_diagnostics(
    model: LocalGlobalScaleDenoiser,
    clean: torch.Tensor,
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    t_grid: list[float],
    diffusion: CoordinateAbsorbingDiffusion,
    seed: int,
) -> dict:
    rows = []
    for index, t_value in enumerate(t_grid):
        generator = torch.Generator(device=clean.device).manual_seed(seed + 7919 * index)
        t = torch.full(
            (len(clean),), float(t_value), device=clean.device, dtype=torch.float32
        )
        noisy, _ = diffusion.corrupt(clean, t, valid_mask, generator)
        _, diagnostic = model(
            noisy, t, coordinates, valid_mask, return_diagnostics=True
        )
        gate = diagnostic["gate"][valid_mask]
        local = diagnostic["local_logits"].permute(0, 2, 3, 1)[valid_mask]
        residual = diagnostic["global_residual"].permute(0, 2, 3, 1)[valid_mask]
        rows.append(
            {
                "t": float(t_value),
                "gate_mean": float(gate.float().mean()),
                "gate_std": float(gate.float().std(unbiased=False)),
                "local_logit_rms": float(local.float().square().mean().sqrt()),
                "global_residual_rms": float(
                    residual.float().square().mean().sqrt()
                ),
            }
        )
    return {"per_t": rows}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variant = str(config["variant"])
    if variant not in STAGE2_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(STAGE2_VARIANTS)}")
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
    log_path = output_dir / "train.jsonl"
    train_parents = load_parent_split(parent_path, "train")
    val_parents = load_parent_split(parent_path, "val")
    if train_parents.lattice_size != val_parents.lattice_size:
        raise ValueError("train and validation parent sizes differ")

    model_config = LocalGlobalDenoiserConfig(**config["model"])
    model = LocalGlobalScaleDenoiser(model_config).to(device)
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
    if bool(config.get("fused_adamw", True)) and device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)

    total_steps = int(args.steps or config["training_steps"])
    widths = [int(value) for value in config["widths"]]
    spacings = [int(value) for value in config["spacings"]]
    batch_sizes = {int(key): int(value) for key, value in config["batch_sizes"].items()}
    accumulation = int(config.get("gradient_accumulation", 2))
    grad_clip = float(config.get("gradient_clip", 1.0))
    ema_decay = float(config.get("ema_decay", 0.999))
    validation_interval = int(config.get("validation_interval", 500))
    checkpoint_interval = int(config.get("checkpoint_interval", 1000))
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

    history: list[dict] = []
    best_validation = float("inf")
    start_step = 0
    elapsed_before_resume = 0.0
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "local_global":
            raise ValueError("resume checkpoint is not a local-global model")
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        history = list(payload.get("history", []))
        best_validation = float(payload.get("best_validation", float("inf")))
        initialization_hash = str(payload["initialization_hash"])
        elapsed_before_resume = float(payload.get("elapsed_seconds", 0.0))
        restore_rng(payload["rng"], training_rng, device_generator)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    header = {
        "event": "start",
        "variant": variant,
        "device": str(device),
        "parameters": parameters,
        "initialization_hash": initialization_hash,
        "parent_data": str(parent_path),
        "train_parents": len(train_parents.spins),
        "val_parents": len(val_parents.spins),
        "start_step": start_step,
        "target_steps": total_steps,
    }
    print(json.dumps(header))
    append_jsonl(log_path, header)
    (output_dir / "run_config.json").write_text(
        json.dumps({**config, "effective_steps": total_steps}, indent=2),
        encoding="utf-8",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for step in range(start_step + 1, total_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_ce = 0.0
        last_metadata: dict = {}
        for _ in range(accumulation):
            clean, coordinates, valid_mask, last_metadata = sample_stage2_batch(
                train_parents.spins,
                variant=variant,
                widths=widths,
                spacings=spacings,
                batch_sizes=batch_sizes,
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
            accumulated_loss += float(result.loss.detach()) / accumulation
            accumulated_ce += float(result.masked_ce.detach()) / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = learning_rate(config, step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        update_ema(ema, model, ema_decay)

        elapsed = elapsed_before_resume + time.perf_counter() - started
        should_validate = step == 1 or step % validation_interval == 0 or step == total_steps
        if should_validate:
            ema.eval()
            validation: dict[str, dict] = {}
            diagnostics: dict[str, dict] = {}
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
                diagnostic_limit = int(config.get("diagnostic_batch_size", 8))
                diagnostics[name] = gate_diagnostics(
                    ema,
                    clean[:diagnostic_limit],
                    coordinates[:diagnostic_limit],
                    valid_mask[:diagnostic_limit],
                    [float(value) for value in config["validation_t_grid"]],
                    diffusion,
                    seed + 30_000,
                )
            anchor_name = str(config.get("anchor_validation", "anchor_w48_s1"))
            anchor_validation = float(validation[anchor_name]["nelbo"])
            record = {
                "event": "validation",
                "step": step,
                "variant": variant,
                **last_metadata,
                "train_loss": accumulated_loss,
                "masked_ce": accumulated_ce,
                "gradient_norm": float(grad_norm),
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "validation": validation,
                "diagnostics": diagnostics,
            }
            history.append(record)
            print(json.dumps(record))
            append_jsonl(log_path, record)
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if anchor_validation < best_validation:
                best_validation = anchor_validation
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        ema,
                        optimizer,
                        config,
                        step,
                        history,
                        best_validation,
                        initialization_hash,
                        training_rng,
                        device_generator,
                        elapsed,
                    ),
                    output_dir / "best_val.pt",
                )
        if step % checkpoint_interval == 0 or step == total_steps:
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    ema,
                    optimizer,
                    config,
                    step,
                    history,
                    best_validation,
                    initialization_hash,
                    training_rng,
                    device_generator,
                    elapsed,
                ),
                last_path,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = elapsed_before_resume + time.perf_counter() - started
    completed_steps = max(total_steps - start_step, 0)
    performance = {
        "elapsed_seconds_total": elapsed,
        "elapsed_seconds_this_run": elapsed - elapsed_before_resume,
        "steps_this_run": completed_steps,
        "steps_per_second_this_run": completed_steps
        / max(elapsed - elapsed_before_resume, 1e-12),
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
    completion = {"event": "complete", "performance": performance}
    append_jsonl(log_path, completion)
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
