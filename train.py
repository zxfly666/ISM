"""Train the absorbing-state discrete diffusion denoiser."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ism_diffusion.diffusion import AbsorbingDiffusion
from ism_diffusion.model import AxialDenoiser, DenoiserConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return project_root / path


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with np.load(path, allow_pickle=False) as data:
        train = torch.from_numpy((data["train"] > 0).astype(np.int64))
        val = torch.from_numpy((data["val"] > 0).astype(np.int64))
        metadata = json.loads(str(data["metadata"].item()))
    return train, val, metadata


def augment(tokens: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """D4 lattice symmetries plus exact h=0 global spin-flip symmetry."""

    batch = len(tokens)
    choices = torch.randint(
        0, 8, (batch,), generator=generator, device=tokens.device
    )
    global_flip = (
        torch.rand(batch, generator=generator, device=tokens.device) < 0.5
    )
    out = torch.empty_like(tokens)
    # Grouping by the eight D4 transformations avoids a GPU synchronization
    # for every sample while producing the same per-sample augmentation family.
    for transform in range(8):
        selected = choices.eq(transform)
        transformed = tokens[selected]
        rotation = transform % 4
        if rotation:
            transformed = torch.rot90(transformed, rotation, dims=(-2, -1))
        if transform >= 4:
            transformed = torch.flip(transformed, dims=(-1,))
        out[selected] = transformed
    return torch.where(global_flip[:, None, None], 1 - out, out)


def learning_rate(config: dict, step: int) -> float:
    peak = float(config["learning_rate"])
    floor = float(config.get("minimum_learning_rate", 0.0))
    warmup = int(config.get("warmup_steps", 0))
    total = int(config["training_steps"])
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
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


def checkpoint_payload(
    model,
    ema,
    optimizer,
    config,
    data_metadata,
    step,
    best_validation,
    history,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "data_metadata": data_metadata,
        "step": step,
        "best_validation": best_validation,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    requested_device = args.device or config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_path = resolve_path(args.config, config["data_path"])
    output_dir = resolve_path(args.config, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tokens, val_tokens, data_metadata = load_data(data_path)
    expected_size = int(data_metadata["lattice_size"])
    if train_tokens.shape[-2:] != (expected_size, expected_size):
        raise ValueError("dataset shape does not match metadata")

    model_config = DenoiserConfig(**config["model"])
    model = AxialDenoiser(model_config).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = AbsorbingDiffusion(
        t_min=float(config["t_min"]),
        t_max=float(config.get("t_max", 1.0)),
        full_mask_probability=float(config.get("full_mask_probability", 0.05)),
    )
    optimizer_options = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(config.get("adam_betas", [0.9, 0.95])),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    fused_adamw = bool(config.get("fused_adamw", False)) and device.type == "cuda"
    if fused_adamw:
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)

    train_model = model
    compile_model = bool(config.get("compile_model", False))
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model requires torch.compile")
        train_model = torch.compile(
            model, mode=str(config.get("compile_mode", "default"))
        )

    start_step = 0
    best_validation = float("inf")
    history: list[dict] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_validation = float(state["best_validation"])
        history = list(state.get("history", []))

    parameters = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameters,
                "train_samples": len(train_tokens),
                "val_samples": len(val_tokens),
                "lattice_size": expected_size,
                "start_step": start_step,
            }
        )
    )

    cpu_generator = torch.Generator().manual_seed(seed + 1)
    device_generator = torch.Generator(device=device).manual_seed(seed + 2)
    preload_data = (
        bool(config.get("preload_data_to_device", False))
        and device.type == "cuda"
    )
    if preload_data:
        train_tokens = train_tokens.to(device)
        val_tokens = val_tokens.to(device)
    batch_size = int(config["batch_size"])
    validation_samples = min(int(config.get("validation_samples", 256)), len(val_tokens))
    fixed_validation = val_tokens[:validation_samples].to(device)
    validation_interval = int(config.get("validation_interval", 1000))
    checkpoint_interval = int(config.get("checkpoint_interval", 5000))
    milestone_steps = {
        int(step) for step in config.get("milestone_steps", [])
    }
    invalid_milestones = {
        step
        for step in milestone_steps
        if step <= 0 or step > int(config["training_steps"])
    }
    if invalid_milestones:
        raise ValueError(
            "milestone_steps must fall within [1, training_steps]: "
            f"{sorted(invalid_milestones)}"
        )
    checkpoint_dir = output_dir / "checkpoints"
    if milestone_steps:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    grad_clip = float(config.get("gradient_clip", 1.0))
    ema_decay = float(config.get("ema_decay", 0.999))
    precision = config.get("precision", "float32")

    if device.type == "cuda" and precision == "bfloat16":
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for step in range(start_step + 1, int(config["training_steps"]) + 1):
        train_model.train()
        selection = torch.randint(
            len(train_tokens),
            (batch_size,),
            generator=cpu_generator,
        )
        if train_tokens.device.type != "cpu":
            selection = selection.to(train_tokens.device, non_blocking=True)
        clean = train_tokens[selection].to(device, non_blocking=True)
        clean = augment(clean, device_generator)
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss_result = diffusion.training_loss(
                train_model, clean, generator=device_generator
            )
        loss_result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        update_ema(ema, model, ema_decay)

        should_validate = step % validation_interval == 0 or step == 1
        if should_validate:
            ema.eval()
            validation, per_t = diffusion.validation_nelbo(
                ema,
                fixed_validation,
                config["validation_t_grid"],
                seed=seed + 10_000,
                batch_size=int(config.get("validation_batch_size", batch_size)),
            )
            record = {
                "step": step,
                "train_loss": float(loss_result.loss.detach().item()),
                "masked_ce": float(loss_result.masked_ce.detach().item()),
                "mask_fraction": float(loss_result.mask_fraction.detach().item()),
                "mean_t": float(loss_result.mean_t.detach().item()),
                "gradient_norm": float(grad_norm),
                "learning_rate": lr,
                "elapsed_seconds": time.perf_counter() - training_started,
                "validation_nelbo": validation,
                "validation_per_t": {str(key): value for key, value in per_t.items()},
            }
            history.append(record)
            print(json.dumps(record))
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if validation < best_validation:
                best_validation = validation
                torch.save(
                    checkpoint_payload(
                        model,
                        ema,
                        optimizer,
                        config,
                        data_metadata,
                        step,
                        best_validation,
                        history,
                    ),
                    output_dir / "best.pt",
                )

        if step % checkpoint_interval == 0 or step == int(config["training_steps"]):
            payload = checkpoint_payload(
                model,
                ema,
                optimizer,
                config,
                data_metadata,
                step,
                best_validation,
                history,
            )
            torch.save(payload, output_dir / "last.pt")
            if step in milestone_steps:
                torch.save(
                    payload,
                    checkpoint_dir / f"step_{step:06d}.pt",
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - training_started
    completed_steps = max(int(config["training_steps"]) - start_step, 1)
    performance = {
        "elapsed_seconds": elapsed,
        "steps_per_second": completed_steps / elapsed,
        "fused_adamw": fused_adamw,
        "preload_data_to_device": preload_data,
        "compile_model": compile_model,
    }
    if device.type == "cuda":
        performance["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        performance["peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    print(json.dumps({"performance": performance}))
    print(f"Training complete. Best validation NELBO: {best_validation:.6f}")


if __name__ == "__main__":
    main()
