"""Paired causal probe for the Stage-2B RandomGap screen.

The probe evaluates identical held-out RandomGap spin fields under matched,
unit/rank, and sample-shuffled coordinates.  All models see the same clean
tokens and corruption masks, so model and coordinate contrasts are paired at
the sample level.  This is a single-training-seed screening experiment, not a
replacement for the later hierarchical multi-seed confirmation.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ism_diffusion.scale_data import coordinate_grid, load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model, write_json
from ism_diffusion.stage2_data import sample_random_gap_windows


def checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_argument, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--t-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95])
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def geometry_schedule() -> list[dict]:
    return [
        {"name": "seen_mix_w64", "width": 64, "gaps": [1, 2, 4, 8]},
        {"name": "held_mix_3_6_w64", "width": 64, "gaps": [3, 6]},
        {"name": "fixed3_w64", "width": 64, "gaps": [3], "fixed_gap": 3},
        {"name": "fixed6_w64", "width": 64, "gaps": [6], "fixed_gap": 6},
        {"name": "fixed10_w48", "width": 48, "gaps": [10], "fixed_gap": 10},
    ]


def inference_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.batch_size < 1:
        raise ValueError("samples must be >=2 and batch-size must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parent_split = load_parent_split(args.parent_data, "test_target")
    parents = parent_split.spins
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

    geometries = geometry_schedule()
    conditions = ("correct", "unit", "shuffled")
    shape = (
        len(models),
        len(geometries),
        len(args.t_grid),
        len(conditions),
        args.samples,
    )
    nll = np.empty(shape, dtype=np.float32)
    brier = np.empty(shape, dtype=np.float32)
    masked_sites = np.empty(
        (len(geometries), len(args.t_grid), args.samples), dtype=np.int32
    )

    cpu = torch.device("cpu")
    for geometry_index, geometry in enumerate(geometries):
        data_rng = np.random.default_rng(args.seed + 1009 * geometry_index)
        clean_cpu, physical_cpu, valid_cpu = sample_random_gap_windows(
            parents,
            width=int(geometry["width"]),
            gaps=list(geometry["gaps"]),
            fixed_gap=geometry.get("fixed_gap"),
            coordinate_mode="matched",
            batch_size=args.samples,
            rng=data_rng,
            device=cpu,
            augment=False,
        )
        width = int(geometry["width"])
        unit_cpu = coordinate_grid(args.samples, width, 1, cpu)
        shuffle_rng = np.random.default_rng(args.seed + 100_003 + geometry_index)
        shuffled_cpu = physical_cpu[torch.from_numpy(shuffle_rng.permutation(args.samples))]
        coordinate_cpu = {
            "correct": physical_cpu,
            "unit": unit_cpu,
            "shuffled": shuffled_cpu,
        }

        for start in range(0, args.samples, args.batch_size):
            stop = min(start + args.batch_size, args.samples)
            clean = clean_cpu[start:stop].to(device=device, non_blocking=True)
            valid = valid_cpu[start:stop].to(device=device, non_blocking=True)
            batch = len(clean)
            coordinate_tensor = torch.cat(
                [coordinate_cpu[name][start:stop] for name in conditions], dim=0
            ).to(device=device, non_blocking=True)
            repeated_clean = clean.repeat((len(conditions), 1, 1))
            repeated_valid = valid.repeat((len(conditions), 1, 1))

            for t_index, t_value in enumerate(args.t_grid):
                mask_rng = np.random.default_rng(
                    args.seed
                    + 1_000_003 * geometry_index
                    + 10_007 * t_index
                    + start
                )
                mask_np = mask_rng.random(clean.shape) < float(t_value)
                mask_np[:, width // 2, width // 2] = True
                masked = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
                noisy = clean.masked_fill(masked, 2)
                masked_sites[geometry_index, t_index, start:stop] = (
                    masked.sum(dim=(1, 2)).cpu().numpy()
                )
                repeated_noisy = noisy.repeat((len(conditions), 1, 1))
                repeated_masked = masked.repeat((len(conditions), 1, 1))
                repeated_t = torch.full(
                    (len(conditions) * batch,), float(t_value), device=device
                )

                for model_index, (_, model) in enumerate(models):
                    with inference_context(device, args.precision):
                        logits = model(
                            repeated_noisy,
                            repeated_t,
                            coordinate_tensor,
                            repeated_valid,
                        )
                    logits = logits.float()
                    ce = F.cross_entropy(logits, repeated_clean, reduction="none")
                    q_plus = F.softmax(logits, dim=1)[:, 1]
                    denominator = repeated_masked.sum(dim=(1, 2)).clamp_min(1)
                    sample_nll = (ce * repeated_masked).sum(dim=(1, 2)) / denominator
                    sample_brier = (
                        (q_plus - repeated_clean.float()).square() * repeated_masked
                    ).sum(dim=(1, 2)) / denominator
                    nll[model_index, geometry_index, t_index, :, start:stop] = (
                        sample_nll.reshape(len(conditions), batch).cpu().numpy()
                    )
                    brier[model_index, geometry_index, t_index, :, start:stop] = (
                        sample_brier.reshape(len(conditions), batch).cpu().numpy()
                    )

        print(
            json.dumps(
                {
                    "event": "geometry_complete",
                    "geometry": geometry["name"],
                    "width": width,
                    "samples": args.samples,
                }
            ),
            flush=True,
        )

    groups = []
    for model_index, (model_name, _) in enumerate(models):
        for geometry_index, geometry in enumerate(geometries):
            for t_index, t_value in enumerate(args.t_grid):
                for condition_index, condition in enumerate(conditions):
                    values = nll[model_index, geometry_index, t_index, condition_index]
                    brier_values = brier[
                        model_index, geometry_index, t_index, condition_index
                    ]
                    groups.append(
                        {
                            "model": model_name,
                            "geometry": geometry["name"],
                            "t": float(t_value),
                            "coordinate_condition": condition,
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
        geometry_names=np.asarray([geometry["name"] for geometry in geometries]),
        geometry_json=np.asarray(json.dumps(geometries)),
        t_grid=np.asarray(args.t_grid),
        coordinate_conditions=np.asarray(conditions),
        nll=nll,
        brier=brier,
        masked_sites=masked_sites,
    )
    summary = {
        "probe": "stage2b_random_gap_causal_screen",
        "scope": "single_training_seed_screen_not_hierarchical_confirmation",
        "settings": {
            "samples": args.samples,
            "batch_size": args.batch_size,
            "t_grid": args.t_grid,
            "seed": args.seed,
            "precision": args.precision,
            "geometries": geometries,
            "coordinate_conditions": list(conditions),
        },
        "models": metadata,
        "groups": groups,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
