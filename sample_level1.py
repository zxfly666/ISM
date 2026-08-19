"""Generate Level-1 samples and compare them with matched open MC crops."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
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
    parser.add_argument(
        "--checkpoint", type=checkpoint_argument, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--reference-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sensitivity-samples", type=int, default=32)
    parser.add_argument("--sensitivity-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampler-method", default="s0", choices=("s0", "s1", "s2"))
    parser.add_argument("--refinement-sweeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def matched_reference_crops(
    parents: np.ndarray, samples: int, width: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parent_indices = rng.integers(len(parents), size=samples)
    origins_x = rng.integers(parents.shape[-1], size=samples)
    origins_y = rng.integers(parents.shape[-1], size=samples)
    return crop_periodic_windows(
        parents,
        parent_indices,
        origins_x,
        origins_y,
        width=width,
        spin_stride=1,
        centered=False,
    )


@torch.inference_mode()
def generate_samples(
    model,
    diffusion: CoordinateAbsorbingDiffusion,
    samples: int,
    width: int,
    steps: int,
    batch_size: int,
    temperature: float,
    seed: int,
    device: torch.device,
    sampler_method: str = "s0",
    refinement_sweeps: int = 2,
) -> tuple[np.ndarray, dict]:
    generated = []
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
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            tokens = sample_with_method(
                model,
                coordinates,
                valid,
                method=sampler_method,
                steps=steps,
                temperature=temperature,
                refinement_sweeps=refinement_sweeps,
                generator=generator,
            )
        generated.append((2 * tokens.cpu().numpy().astype(np.int8) - 1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    performance = {
        "samples": samples,
        "steps": steps,
        "method": sampler_method,
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
    return np.concatenate(generated, axis=0), performance


def metric_bundle(spins: np.ndarray, max_radius: int) -> dict:
    summary, radii, raw, connected = open_ensemble_metrics(
        spins, max_radius=max_radius
    )
    return {
        "summary": summary,
        "radii": radii,
        "raw": raw,
        "connected": connected,
    }


def scalar_difference(left: dict, right: dict) -> dict[str, float]:
    keys = (
        "energy_mean",
        "magnetization_mean",
        "abs_magnetization_mean",
        "low_frequency_power_mean",
    )
    return {key: float(left[key] - right[key]) for key in keys}


def plot_results(
    output: Path,
    models: dict[str, np.ndarray],
    target: np.ndarray,
    bundles: dict[str, dict],
) -> None:
    names = ["MC target", *models]
    arrays = [target, *models.values()]
    figure, axes = plt.subplots(len(names), 8, figsize=(12, 1.55 * len(names)))
    axes = np.atleast_2d(axes)
    for row, (name, values) in enumerate(zip(names, arrays)):
        for column in range(8):
            if column < len(values):
                axes[row, column].imshow(
                    values[column], cmap="coolwarm", vmin=-1, vmax=1
                )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column >= len(values):
                axes[row, column].set_visible(False)
            if column == 0:
                axes[row, column].set_ylabel(name)
    figure.tight_layout()
    figure.savefig(output / "sample_grid.png", dpi=180)
    figure.savefig(output / "sample_grid.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, bundle in bundles.items():
        axes[0].plot(bundle["radii"][1:], bundle["raw"][1:], label=name)
        axes[1].plot(bundle["radii"][1:], bundle["connected"][1:], label=name)
    axes[0].set_title("Raw open-window correlation")
    axes[1].set_title("Ensemble-connected correlation")
    for axis in axes:
        axis.set_xlabel("distance r")
        axis.set_ylabel("G(r)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "correlation.png", dpi=180)
    figure.savefig(output / "correlation.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_parents = load_parent_split(args.parent_data, "test_target").spins
    control_parents = load_parent_split(args.parent_data, "test_control").spins
    target = matched_reference_crops(
        target_parents, args.reference_samples, args.width, args.seed + 10
    )
    control = matched_reference_crops(
        control_parents, args.reference_samples, args.width, args.seed + 20
    )
    max_radius = args.width // 2
    bundles = {
        "MC target": metric_bundle(target, max_radius),
        "MC control": metric_bundle(control, max_radius),
    }
    generated: dict[str, np.ndarray] = {}
    sensitivity: dict[str, np.ndarray] = {}
    report = {
        "settings": vars(args) | {
            "parent_data": str(args.parent_data),
            "output_dir": str(args.output_dir),
            "checkpoint": [(name, str(path)) for name, path in args.checkpoint],
            "device": str(device),
        },
        "reference": {
            "target": bundles["MC target"]["summary"],
            "control": bundles["MC control"]["summary"],
            "target_control_scalar_difference": scalar_difference(
                bundles["MC control"]["summary"], bundles["MC target"]["summary"]
            ),
            "target_control_correlation_error": correlation_band_errors(
                bundles["MC control"]["connected"], bundles["MC target"]["connected"]
            ),
        },
        "models": {},
    }

    for model_index, (name, checkpoint) in enumerate(args.checkpoint):
        model, payload = load_scale_model(checkpoint, device)
        config = payload["config"]
        diffusion = CoordinateAbsorbingDiffusion(
            t_min=float(config.get("t_min", 0.01)),
            t_max=float(config.get("t_max", 1.0)),
            full_mask_probability=float(config.get("full_mask_probability", 0.02)),
        )
        values, performance = generate_samples(
            model,
            diffusion,
            args.samples,
            args.width,
            args.steps,
            args.batch_size,
            args.temperature,
            args.seed + 1000 * model_index,
            device,
            args.sampler_method,
            args.refinement_sweeps,
        )
        if args.sensitivity_samples > 0:
            sensitivity_values, sensitivity_performance = generate_samples(
                model,
                diffusion,
                args.sensitivity_samples,
                args.width,
                args.sensitivity_steps,
                args.batch_size,
                args.temperature,
                args.seed + 1000 * model_index + 500,
                device,
                args.sampler_method,
                args.refinement_sweeps,
            )
        else:
            sensitivity_values = np.empty((0, args.width, args.width), dtype=np.int8)
            sensitivity_performance = None
        generated[name] = values
        sensitivity[name] = sensitivity_values
        bundles[name] = metric_bundle(values, max_radius)
        sensitivity_bundle = (
            metric_bundle(sensitivity_values, max_radius)
            if len(sensitivity_values)
            else None
        )
        report["models"][name] = {
            "checkpoint": str(checkpoint),
            "step": int(payload["step"]),
            "variant": config["variant"],
            "initialization_hash": payload.get("initialization_hash"),
            "metrics": bundles[name]["summary"],
            "scalar_difference_from_target": scalar_difference(
                bundles[name]["summary"], bundles["MC target"]["summary"]
            ),
            "connected_correlation_error": correlation_band_errors(
                bundles[name]["connected"], bundles["MC target"]["connected"]
            ),
            "sampler_performance": performance,
            "sensitivity_128": (
                {
                    "metrics": sensitivity_bundle["summary"],
                    "difference_from_64": scalar_difference(
                        sensitivity_bundle["summary"], bundles[name]["summary"]
                    ),
                    "connected_correlation_difference": correlation_band_errors(
                        sensitivity_bundle["connected"], bundles[name]["connected"]
                    ),
                    "performance": sensitivity_performance,
                }
                if sensitivity_bundle is not None
                else None
            ),
        }

    arrays = {
        "mc_target": target,
        "mc_control": control,
    }
    for name, values in generated.items():
        arrays[f"model_{name}"] = values
        arrays[f"sensitivity_{name}"] = sensitivity[name]
    for name, bundle in bundles.items():
        safe = name.lower().replace(" ", "_")
        arrays[f"radii_{safe}"] = bundle["radii"]
        arrays[f"raw_{safe}"] = bundle["raw"]
        arrays[f"connected_{safe}"] = bundle["connected"]
    np.savez_compressed(args.output_dir / "samples_and_correlations.npz", **arrays)
    write_json(args.output_dir / "metrics.json", report)
    plot_results(args.output_dir, generated, target, bundles)


if __name__ == "__main__":
    main()
