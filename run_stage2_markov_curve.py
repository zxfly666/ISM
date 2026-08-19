"""Run the exact Markov contamination probe over multiple context widths."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from probe_level1 import checkpoint_argument, run_markov


def width_count(value: str) -> tuple[int, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("width/count must be W=N")
    width, count = value.split("=", 1)
    return int(width), int(count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_argument, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--width-count",
        action="append",
        type=width_count,
        default=None,
        help="repeat W=N; defaults to a compute-balanced Stage-2A curve",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--t-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.95])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = args.width_count or [
        (8, 2048),
        (16, 2048),
        (24, 2048),
        (32, 1024),
        (48, 512),
        (64, 256),
    ]
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    for index, (width, centers) in enumerate(schedule):
        namespace = argparse.Namespace(
            parent_data=args.parent_data,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir / f"w{width:03d}",
            precision=args.precision,
            centers=int(centers),
            large_width=int(width),
            batch_size=args.batch_size,
            seed=args.seed + 1009 * index,
            t_grid=args.t_grid,
        )
        run_markov(namespace, device)
        print(f"completed Markov width={width} centers={centers}", flush=True)


if __name__ == "__main__":
    main()
