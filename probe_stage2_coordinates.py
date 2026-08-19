"""Paired continuous coordinate-response curves for Stage 2A."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import (
    centered_coordinate_grid,
    crop_periodic_windows,
    load_scale_model,
    spin_tokens,
    write_json,
)


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
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--physical-strides", type=int, nargs="+", default=[3, 6])
    parser.add_argument(
        "--coordinate-scales",
        type=float,
        nargs="+",
        default=[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10],
    )
    parser.add_argument("--t-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents = load_parent_split(args.parent_data, "test_target").spins
    models = []
    metadata = {}
    for name, path in args.checkpoint:
        model, payload = load_scale_model(path, device)
        models.append((name, model))
        metadata[name] = {
            "checkpoint": str(path),
            "variant": payload["config"]["variant"],
            "step": int(payload["step"]),
            "initialization_hash": payload.get("initialization_hash"),
        }

    shape = (
        len(models),
        len(args.physical_strides),
        len(args.t_grid),
        len(args.coordinate_scales),
        args.samples,
    )
    nll = np.empty(shape, dtype=np.float32)
    brier = np.empty(shape, dtype=np.float32)
    masked_sites = np.empty(
        (len(args.physical_strides), len(args.t_grid), args.samples),
        dtype=np.int32,
    )

    for stride_index, physical_stride in enumerate(args.physical_strides):
        rng = np.random.default_rng(args.seed + 1009 * stride_index)
        parent_indices = rng.integers(len(parents), size=args.samples)
        origins_x = rng.integers(parents.shape[-1], size=args.samples)
        origins_y = rng.integers(parents.shape[-1], size=args.samples)
        windows = crop_periodic_windows(
            parents,
            parent_indices,
            origins_x,
            origins_y,
            width=args.width,
            spin_stride=physical_stride,
            centered=False,
        )
        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            clean = spin_tokens(windows[start:stop], device)
            valid = torch.ones_like(clean, dtype=torch.bool)
            batch = len(clean)
            for t_index, t_value in enumerate(args.t_grid):
                mask_rng = np.random.default_rng(
                    args.seed + 1_000_003 * stride_index + 10_007 * t_index + start
                )
                mask_np = mask_rng.random(clean.shape) < float(t_value)
                mask_np[:, args.width // 2, args.width // 2] = True
                masked = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
                noisy = clean.masked_fill(masked, 2)
                masked_sites[stride_index, t_index, start:stop] = (
                    masked.sum(dim=(1, 2)).cpu().numpy()
                )

                coordinate_batches = [
                    centered_coordinate_grid(batch, args.width, scale, device)
                    for scale in args.coordinate_scales
                ]
                coordinate_tensor = torch.cat(coordinate_batches, dim=0)
                repeat = len(args.coordinate_scales)
                repeated_noisy = noisy.repeat((repeat, 1, 1))
                repeated_clean = clean.repeat((repeat, 1, 1))
                repeated_valid = valid.repeat((repeat, 1, 1))
                repeated_masked = masked.repeat((repeat, 1, 1))
                repeated_t = torch.full(
                    (repeat * batch,), float(t_value), device=device
                )
                for model_index, (_, model) in enumerate(models):
                    context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and args.precision == "bfloat16"
                        else nullcontext()
                    )
                    with context:
                        logits = model(
                            repeated_noisy,
                            repeated_t,
                            coordinate_tensor,
                            repeated_valid,
                        )
                    ce = F.cross_entropy(
                        logits.float(), repeated_clean, reduction="none"
                    )
                    q_plus = F.softmax(logits.float(), dim=1)[:, 1]
                    sample_nll = (ce * repeated_masked).sum(dim=(1, 2)) / repeated_masked.sum(
                        dim=(1, 2)
                    )
                    sample_brier = (
                        (q_plus - repeated_clean.float()).square() * repeated_masked
                    ).sum(dim=(1, 2)) / repeated_masked.sum(dim=(1, 2))
                    nll[model_index, stride_index, t_index, :, start:stop] = (
                        sample_nll.reshape(repeat, batch).cpu().numpy()
                    )
                    brier[model_index, stride_index, t_index, :, start:stop] = (
                        sample_brier.reshape(repeat, batch).cpu().numpy()
                    )

    groups = []
    response = []
    for model_index, (name, _) in enumerate(models):
        for stride_index, physical_stride in enumerate(args.physical_strides):
            averaged_over_t = nll[model_index, stride_index].mean(axis=0).mean(axis=-1)
            best_index = int(np.argmin(averaged_over_t))
            response.append(
                {
                    "model": name,
                    "physical_stride": int(physical_stride),
                    "best_coordinate_scale": float(args.coordinate_scales[best_index]),
                    "best_nll": float(averaged_over_t[best_index]),
                    "unit_regret": float(averaged_over_t[0] - averaged_over_t[best_index]),
                }
            )
            for t_index, t_value in enumerate(args.t_grid):
                for scale_index, scale in enumerate(args.coordinate_scales):
                    values = nll[model_index, stride_index, t_index, scale_index]
                    brier_values = brier[model_index, stride_index, t_index, scale_index]
                    groups.append(
                        {
                            "model": name,
                            "physical_stride": int(physical_stride),
                            "t": float(t_value),
                            "coordinate_scale": float(scale),
                            "samples": args.samples,
                            "nll_mean": float(values.mean()),
                            "nll_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                            "brier_mean": float(brier_values.mean()),
                            "brier_se": float(
                                brier_values.std(ddof=1) / np.sqrt(len(brier_values))
                            ),
                        }
                    )
    np.savez_compressed(
        args.output_dir / "per_sample.npz",
        model_names=np.asarray([name for name, _ in models]),
        physical_strides=np.asarray(args.physical_strides),
        t_grid=np.asarray(args.t_grid),
        coordinate_scales=np.asarray(args.coordinate_scales),
        nll=nll,
        brier=brier,
        masked_sites=masked_sites,
    )
    summary = {
        "probe": "continuous_coordinate_response",
        "settings": {
            "samples": args.samples,
            "width": args.width,
            "physical_strides": args.physical_strides,
            "coordinate_scales": args.coordinate_scales,
            "t_grid": args.t_grid,
            "precision": args.precision,
            "seed": args.seed,
        },
        "models": metadata,
        "response": response,
        "groups": groups,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", "response": response}))


if __name__ == "__main__":
    main()
