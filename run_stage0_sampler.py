"""Run resumable Stage-0 sampler closure on the validation MC split."""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import (
    centered_coordinate_grid,
    correlation_band_errors,
    crop_periodic_windows,
    load_scale_model,
    open_ensemble_metrics,
    write_json,
)
from ism_diffusion.stage2_sampling import sample_with_method


def checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=checkpoint_argument, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--reference-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=["s0", "s1", "s2"])
    parser.add_argument("--steps", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--refinement-sweeps", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def reference_crops(parents: np.ndarray, samples: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(parents), size=samples)
    origins_x = rng.integers(parents.shape[-1], size=samples)
    origins_y = rng.integers(parents.shape[-1], size=samples)
    return crop_periodic_windows(
        parents,
        indices,
        origins_x,
        origins_y,
        width=width,
        spin_stride=1,
        centered=False,
    )


@torch.inference_mode()
def generate(
    model,
    samples: int,
    width: int,
    batch_size: int,
    method: str,
    steps: int,
    temperature: float,
    refinement_sweeps: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, dict]:
    values = []
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, samples, batch_size):
        batch = min(batch_size, samples - start)
        coordinates = centered_coordinate_grid(batch, width, 1.0, device)
        valid = torch.ones((batch, width, width), dtype=torch.bool, device=device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bfloat16"
            else nullcontext()
        )
        with context:
            tokens = sample_with_method(
                model,
                coordinates,
                valid,
                method=method,
                steps=steps,
                temperature=temperature,
                refinement_sweeps=refinement_sweeps,
                generator=generator,
            )
        values.append(2 * tokens.cpu().numpy().astype(np.int8) - 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    performance = {
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / samples,
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
    return np.concatenate(values), performance


def bundle(values: np.ndarray, max_radius: int) -> tuple[dict, np.ndarray]:
    summary, _, _, connected = open_ensemble_metrics(values, max_radius=max_radius)
    return summary, connected


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    validation = load_parent_split(args.parent_data, "val").spins
    target = reference_crops(validation, args.reference_samples, args.width, 901)
    control = reference_crops(validation, args.reference_samples, args.width, 1901)
    target_summary, target_connected = bundle(target, args.width // 2)
    control_summary, control_connected = bundle(control, args.width // 2)
    np.savez_compressed(args.output_dir / "validation_reference.npz", target=target, control=control)
    write_json(
        args.output_dir / "reference_metrics.json",
        {
            "target": target_summary,
            "control": control_summary,
            "target_control_correlation": correlation_band_errors(
                control_connected, target_connected
            ),
        },
    )

    manifest = {
        "settings": {
            **vars(args),
            "parent_data": str(args.parent_data),
            "output_dir": str(args.output_dir),
            "checkpoint": [(name, str(path)) for name, path in args.checkpoint],
            "device": str(device),
        },
        "reference": {
            "target": target_summary,
            "control": control_summary,
        },
        "records": [],
    }
    for model_index, (model_name, checkpoint) in enumerate(args.checkpoint):
        model, payload = load_scale_model(checkpoint, device)
        for method_index, method in enumerate(args.methods):
            for steps in args.steps:
                for seed in args.seeds:
                    stem = "__".join(
                        (safe_name(model_name), safe_name(method), f"n{steps}", f"s{seed}")
                    )
                    data_path = runs_dir / f"{stem}.npz"
                    metrics_path = runs_dir / f"{stem}.json"
                    if data_path.exists() and metrics_path.exists():
                        manifest["records"].append(
                            json.loads(metrics_path.read_text(encoding="utf-8"))
                        )
                        continue
                    run_seed = int(seed + 100_003 * model_index + 10_007 * method_index)
                    generated, performance = generate(
                        model,
                        samples=args.samples,
                        width=args.width,
                        batch_size=args.batch_size,
                        method=method,
                        steps=steps,
                        temperature=args.temperature,
                        refinement_sweeps=args.refinement_sweeps,
                        seed=run_seed,
                        device=device,
                        precision=args.precision,
                    )
                    summary, connected = bundle(generated, args.width // 2)
                    record = {
                        "model": model_name,
                        "variant": payload["config"]["variant"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_step": int(payload["step"]),
                        "method": method,
                        "steps": int(steps),
                        "seed": int(seed),
                        "run_seed": run_seed,
                        "samples": args.samples,
                        "metrics": summary,
                        "correlation_error": correlation_band_errors(
                            connected, target_connected
                        ),
                        "performance": performance,
                    }
                    np.savez_compressed(
                        data_path,
                        generated=generated,
                        connected=connected,
                        target_connected=target_connected,
                    )
                    write_json(metrics_path, record)
                    manifest["records"].append(record)
                    write_json(args.output_dir / "manifest.json", manifest)
                    print(json.dumps({"event": "completed", **record}))
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
