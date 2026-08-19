"""Run the two mechanism probes for the Level-1 scale-aware pilot."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ism_diffusion.ising import BETA_CRITICAL
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
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=checkpoint_argument, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    subparsers = parser.add_subparsers(dest="probe", required=True)

    markov = subparsers.add_parser("markov")
    markov.add_argument("--centers", type=int, default=512)
    markov.add_argument("--large-width", type=int, default=48)
    markov.add_argument("--batch-size", type=int, default=4)
    markov.add_argument("--seed", type=int, default=314159)
    markov.add_argument("--t-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95])

    coordinates = subparsers.add_parser("coordinates")
    coordinates.add_argument("--samples", type=int, default=512)
    coordinates.add_argument("--width", type=int, default=24)
    coordinates.add_argument("--strides", type=int, nargs="+", default=[3, 6])
    coordinates.add_argument("--batch-size", type=int, default=8)
    coordinates.add_argument("--wrong-scale", type=float, default=0.5)
    coordinates.add_argument("--seed", type=int, default=271828)
    coordinates.add_argument(
        "--t-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95]
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def bernoulli_metrics(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, ...]:
    epsilon = 1e-7
    q = np.clip(q, epsilon, 1.0 - epsilon)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    soft_ce = -(p * np.log(q) + (1.0 - p) * np.log(1.0 - q))
    brier = (q - p) ** 2
    logit = np.log(q / (1.0 - q))
    return kl, soft_ce, brier, logit


@torch.inference_mode()
def plus_probability(
    model,
    tokens: torch.Tensor,
    t_value: float,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    row: int,
    column: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    t = torch.full((len(tokens),), float(t_value), device=device)
    with autocast_context(device, precision):
        logits = model(tokens, t, coordinates, valid)
    probability = F.softmax(logits.float(), dim=1)[:, 1, row, column]
    return probability.cpu().numpy()


def summarize_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        group = tuple(row[key] for key in keys)
        for metric in ("kl", "soft_ce", "brier", "nll"):
            if metric in row:
                groups[group][metric].append(float(row[metric]))
        if "logit_drift" in row:
            groups[group]["logit_drift_abs"].append(abs(float(row["logit_drift"])))
    output = []
    for group, metrics in sorted(groups.items()):
        record = {key: value for key, value in zip(keys, group)}
        record["samples"] = len(next(iter(metrics.values())))
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            record[f"{metric}_mean"] = float(array.mean())
            record[f"{metric}_se"] = (
                float(array.std(ddof=1) / math.sqrt(len(array)))
                if len(array) > 1
                else 0.0
            )
        output.append(record)
    return {"groups": output}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_markov(args: argparse.Namespace, device: torch.device) -> None:
    parents = load_parent_split(args.parent_data, "test_target").spins
    rng = np.random.default_rng(args.seed)
    parent_indices = rng.integers(len(parents), size=args.centers)
    origins_x = rng.integers(parents.shape[-1], size=args.centers)
    origins_y = rng.integers(parents.shape[-1], size=args.centers)
    rows: list[dict] = []

    models = {}
    metadata = {}
    for name, path in args.checkpoint:
        model, payload = load_scale_model(path, device)
        models[name] = model
        metadata[name] = {
            "checkpoint": str(path),
            "step": int(payload["step"]),
            "variant": payload["config"]["variant"],
            "initialization_hash": payload.get("initialization_hash"),
        }

    for start in range(0, args.centers, args.batch_size):
        stop = min(start + args.batch_size, args.centers)
        selected = slice(start, stop)
        patch = crop_periodic_windows(
            parents,
            parent_indices[selected],
            origins_x[selected],
            origins_y[selected],
            width=args.large_width,
            spin_stride=1,
            centered=True,
        )
        batch = len(patch)
        center = args.large_width // 2
        large_clean = spin_tokens(patch, device)
        neighbour_values = np.stack(
            (
                patch[:, center - 1, center],
                patch[:, center + 1, center],
                patch[:, center, center - 1],
                patch[:, center, center + 1],
            ),
            axis=1,
        )
        exact = 1.0 / (
            1.0 + np.exp(-2.0 * float(BETA_CRITICAL) * neighbour_values.sum(axis=1))
        )

        small_tokens = torch.full((batch, 3, 3), 3, dtype=torch.long, device=device)
        small_valid = torch.zeros((batch, 3, 3), dtype=torch.bool, device=device)
        large_pad_tokens = torch.full_like(large_clean, 3)
        large_pad_valid = torch.zeros_like(large_clean, dtype=torch.bool)
        cross = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for dx, dy in cross:
            values = large_clean[:, center + dx, center + dy]
            small_tokens[:, 1 + dx, 1 + dy] = values
            small_valid[:, 1 + dx, 1 + dy] = True
            large_pad_tokens[:, center + dx, center + dy] = values
            large_pad_valid[:, center + dx, center + dy] = True
        small_tokens[:, 1, 1] = 2
        small_valid[:, 1, 1] = True
        large_pad_tokens[:, center, center] = 2
        large_pad_valid[:, center, center] = True

        large_mask_tokens = torch.full_like(large_clean, 2)
        for dx, dy in cross:
            large_mask_tokens[:, center + dx, center + dy] = large_clean[
                :, center + dx, center + dy
            ]
        large_visible_tokens = large_clean.clone()
        large_visible_tokens[:, center, center] = 2
        large_valid = torch.ones_like(large_clean, dtype=torch.bool)
        contexts = {
            "Small": (
                small_tokens,
                centered_coordinate_grid(batch, 3, 1.0, device),
                small_valid,
                1,
                1,
            ),
            "Large-PAD": (
                large_pad_tokens,
                centered_coordinate_grid(batch, args.large_width, 1.0, device),
                large_pad_valid,
                center,
                center,
            ),
            "Large-MASK": (
                large_mask_tokens,
                centered_coordinate_grid(batch, args.large_width, 1.0, device),
                large_valid,
                center,
                center,
            ),
            "Large-Visible": (
                large_visible_tokens,
                centered_coordinate_grid(batch, args.large_width, 1.0, device),
                large_valid,
                center,
                center,
            ),
        }

        for model_name, model in models.items():
            for t_value in args.t_grid:
                probabilities = {}

                # ``Small`` has a different spatial shape, so it remains one
                # forward pass.  The three large contexts share their shape
                # and coordinates; concatenate them along the batch axis to
                # replace three small GPU launches with one larger launch.
                # This is numerically equivalent in eval/inference mode and
                # materially improves accelerator occupancy for the formal
                # Markov probe.
                small_tokens_, small_coordinates, small_valid_, row, column = (
                    contexts["Small"]
                )
                probabilities["Small"] = plus_probability(
                    model,
                    small_tokens_,
                    t_value,
                    small_coordinates,
                    small_valid_,
                    row,
                    column,
                    device,
                    args.precision,
                )

                large_names = ("Large-PAD", "Large-MASK", "Large-Visible")
                large_tokens_ = torch.cat(
                    [contexts[name][0] for name in large_names], dim=0
                )
                large_coordinates = torch.cat(
                    [contexts[name][1] for name in large_names], dim=0
                )
                large_valid_ = torch.cat(
                    [contexts[name][2] for name in large_names], dim=0
                )
                large_probability = plus_probability(
                    model,
                    large_tokens_,
                    t_value,
                    large_coordinates,
                    large_valid_,
                    center,
                    center,
                    device,
                    args.precision,
                )
                for context_index, context_name in enumerate(large_names):
                    left = context_index * batch
                    probabilities[context_name] = large_probability[left : left + batch]

                logits = {
                    context_name: np.log(
                        np.clip(probability, 1e-7, 1 - 1e-7)
                        / np.clip(1 - probability, 1e-7, 1 - 1e-7)
                    )
                    for context_name, probability in probabilities.items()
                }
                for context_name, probability in probabilities.items():
                    kl, soft_ce, brier, logit = bernoulli_metrics(exact, probability)
                    for local_index in range(batch):
                        rows.append(
                            {
                                "sample_id": start + local_index,
                                "model": model_name,
                                "t": float(t_value),
                                "context": context_name,
                                "p_exact_plus": float(exact[local_index]),
                                "p_model_plus": float(probability[local_index]),
                                "kl": float(kl[local_index]),
                                "soft_ce": float(soft_ce[local_index]),
                                "brier": float(brier[local_index]),
                                "logit": float(logit[local_index]),
                                "logit_drift": float(
                                    logits[context_name][local_index]
                                    - logits["Small"][local_index]
                                ),
                            }
                        )

    output = args.output_dir / "markov"
    write_csv(output / "per_sample.csv", rows)
    summary = summarize_rows(rows, ("model", "t", "context"))
    summary.update(
        {
            "probe": "exact_markov_blanket_contamination",
            "centers": args.centers,
            "large_width": args.large_width,
            "models": metadata,
            "precision": args.precision,
        }
    )
    write_json(output / "summary.json", summary)


def coordinate_conditions(name: str, stride: int, wrong_scale: float) -> list[tuple[str, float]]:
    normalized = name.lower()
    if normalized == "t3" or normalized.endswith("-t3") or normalized.endswith("_t3"):
        return [
            ("CorrectCoord", float(stride)),
            ("UnitCoord", 1.0),
            ("WrongScale", float(stride) * wrong_scale),
        ]
    if normalized == "punit" or normalized.endswith("-punit") or normalized.endswith("_punit"):
        return [("UnitCoord", 1.0)]
    if normalized == "pphase":
        return [("PhaseCoord", float(stride))]
    return [("ConfiguredCoord", float(stride))]


def run_coordinates(args: argparse.Namespace, device: torch.device) -> None:
    parents = load_parent_split(args.parent_data, "test_target").spins
    models = {}
    metadata = {}
    for name, path in args.checkpoint:
        model, payload = load_scale_model(path, device)
        models[name] = model
        metadata[name] = {
            "checkpoint": str(path),
            "step": int(payload["step"]),
            "variant": payload["config"]["variant"],
            "initialization_hash": payload.get("initialization_hash"),
        }

    rows: list[dict] = []
    for stride_index, stride in enumerate(args.strides):
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
            spin_stride=stride,
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
                labels = clean
                for model_name, model in models.items():
                    for condition, coordinate_stride in coordinate_conditions(
                        model_name, stride, args.wrong_scale
                    ):
                        coordinates = centered_coordinate_grid(
                            batch, args.width, coordinate_stride, device
                        )
                        t = torch.full((batch,), float(t_value), device=device)
                        with autocast_context(device, args.precision):
                            logits = model(noisy, t, coordinates, valid)
                        ce = F.cross_entropy(logits.float(), labels, reduction="none")
                        q_plus = F.softmax(logits.float(), dim=1)[:, 1]
                        sample_nll = (ce * masked).sum(dim=(1, 2)) / masked.sum(
                            dim=(1, 2)
                        )
                        sample_brier = (
                            (q_plus - labels.float()).square() * masked
                        ).sum(dim=(1, 2)) / masked.sum(dim=(1, 2))
                        for local_index in range(batch):
                            rows.append(
                                {
                                    "sample_id": start + local_index,
                                    "model": model_name,
                                    "physical_stride": int(stride),
                                    "t": float(t_value),
                                    "condition": condition,
                                    "coordinate_stride": float(coordinate_stride),
                                    "masked_sites": int(masked[local_index].sum().item()),
                                    "nll": float(sample_nll[local_index].item()),
                                    "brier": float(sample_brier[local_index].item()),
                                }
                            )

    output = args.output_dir / "coordinates"
    write_csv(output / "per_sample.csv", rows)
    summary = summarize_rows(
        rows, ("model", "physical_stride", "t", "condition")
    )
    summary.update(
        {
            "probe": "paired_coordinate_causal_test",
            "samples_per_stride": args.samples,
            "width": args.width,
            "strides": args.strides,
            "wrong_scale_multiplier": args.wrong_scale,
            "models": metadata,
            "precision": args.precision,
        }
    )
    write_json(output / "summary.json", summary)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probe == "markov":
        run_markov(args, device)
    else:
        run_coordinates(args, device)


if __name__ == "__main__":
    main()
