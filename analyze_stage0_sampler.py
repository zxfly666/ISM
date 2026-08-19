"""Select and freeze one common sampler using Stage-0 validation results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ism_diffusion.scale_evaluation import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = payload["reference"]["target"]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in payload["records"]:
        grouped[(str(record["method"]), int(record["steps"]))].append(record)

    candidates = []
    for (method, steps), records in sorted(grouped.items()):
        short = np.asarray(
            [row["correlation_error"]["short"]["nrmse"] for row in records]
        )
        medium = np.asarray(
            [
                row["correlation_error"].get("medium", row["correlation_error"]["short"])[
                    "nrmse"
                ]
                for row in records
            ]
        )
        expanded = np.asarray(
            [
                row["correlation_error"].get(
                    "expanded",
                    row["correlation_error"].get(
                        "medium", row["correlation_error"]["short"]
                    ),
                )["nrmse"]
                for row in records
            ]
        )
        energy = np.asarray(
            [
                abs(row["metrics"]["energy_mean"] - reference["energy_mean"])
                for row in records
            ]
        )
        abs_magnetization = np.asarray(
            [
                abs(
                    row["metrics"]["abs_magnetization_mean"]
                    - reference["abs_magnetization_mean"]
                )
                for row in records
            ]
        )
        # Fixed before looking at results: correlation carries 75% of the
        # score; energy and |m| guard obvious local/global failure.
        score_rows = (
            0.25 * short
            + 0.20 * medium
            + 0.30 * expanded
            + 0.15 * energy / 0.05
            + 0.10 * abs_magnetization / 0.05
        )
        candidates.append(
            {
                "method": method,
                "steps": steps,
                "records": len(records),
                "models": sorted({row["model"] for row in records}),
                "seeds": sorted({int(row["seed"]) for row in records}),
                "score_mean": float(score_rows.mean()),
                "score_se": float(score_rows.std(ddof=1) / np.sqrt(len(score_rows)))
                if len(score_rows) > 1
                else 0.0,
                "short_nrmse_mean": float(short.mean()),
                "medium_nrmse_mean": float(medium.mean()),
                "expanded_nrmse_mean": float(expanded.mean()),
                "energy_abs_error_mean": float(energy.mean()),
                "abs_magnetization_error_mean": float(abs_magnetization.mean()),
            }
        )
    if not candidates:
        raise ValueError("manifest contains no completed sampler records")
    ranked = sorted(candidates, key=lambda row: (row["score_mean"], row["steps"]))
    winner = ranked[0]
    output = {
        "status": "FROZEN",
        "selection_split": "validation",
        "selection_rule": "minimum preregistered composite score; lower steps break ties",
        "frozen_sampler": {
            "method": winner["method"],
            "steps": winner["steps"],
            "temperature": payload["settings"]["temperature"],
            "refinement_sweeps": payload["settings"]["refinement_sweeps"],
        },
        "ranking": ranked,
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
