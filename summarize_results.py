"""Print a compact table from one or more sampling metrics.json files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+")
    args = parser.parse_args()
    paths: set[Path] = set()
    for pattern in args.patterns:
        paths.update(Path(path) for path in glob.glob(pattern))

    header = ("run", "seed", "sampler", "energy", "abs_m", "binder", "xi_l", "sum_abs")
    print("\t".join(header))
    rows: list[tuple] = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        model = metrics["model"]
        sampling = metrics.get("sampling", {})
        rows.append(
            (
                path.parent.name,
                sampling.get("seed", ""),
                sampling.get("sampler", ""),
                model["energy_mean"],
                model["abs_magnetization_mean"],
                model["binder_u4"],
                model["xi_over_l"],
                sum(metrics["absolute_error"].values()),
            )
        )
    for row in sorted(rows, key=lambda item: item[-1]):
        print(
            f"{row[0]}\t{row[1]}\t{row[2]}\t"
            f"{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}\t"
            f"{row[6]:.6f}\t{row[7]:.6f}"
        )


if __name__ == "__main__":
    main()
