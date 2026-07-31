"""Generate configurations and compare their ensemble physics with MC reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ism_diffusion.diffusion import AbsorbingDiffusion
from ism_diffusion.ising import BETA_CRITICAL, generate_independent_chains
from ism_diffusion.metrics import (
    compare_ensembles,
    correlation_map,
    radial_average_periodic,
    structure_factor,
)
from ism_diffusion.model import AxialDenoiser, DenoiserConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--sampler",
        choices=(
            "ancestral",
            "ancestral_corrector",
            "confidence",
            "confidence_corrector",
        ),
        default="ancestral",
    )
    parser.add_argument("--corrector-steps", type=int, default=4)
    parser.add_argument("--corrector-mask-ratio", type=float, default=0.1)
    parser.add_argument("--selection-noise", type=float, default=0.05)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument(
        "--reference-split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--mc-chains", type=int, default=4)
    parser.add_argument("--mc-burn-in-sweeps", type=int, default=500)
    parser.add_argument("--mc-sweeps-between", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_reference(args: argparse.Namespace) -> np.ndarray:
    if args.reference_data:
        with np.load(args.reference_data, allow_pickle=False) as data:
            if args.reference_split not in data.files:
                raise ValueError(
                    f"reference-data has no {args.reference_split!r} split"
                )
            reference = data[args.reference_split].astype(np.int8)
        if reference.shape[-1] != args.lattice_size:
            raise ValueError("reference-data lattice size does not match request")
        return reference[: args.samples]
    per_chain = int(np.ceil(args.samples / args.mc_chains))
    reference, _ = generate_independent_chains(
        lattice_size=args.lattice_size,
        chain_seeds=[args.seed + 100 + i for i in range(args.mc_chains)],
        samples_per_chain=per_chain,
        burn_in_sweeps=args.mc_burn_in_sweeps,
        sweeps_between=args.mc_sweeps_between,
        beta=float(BETA_CRITICAL),
    )
    return reference[: args.samples]


def save_figure(
    output: Path, model_spins: np.ndarray, reference_spins: np.ndarray
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(model_spins[0], cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 0].set_title("Model sample")
    axes[1, 0].imshow(reference_spins[0], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_title("MC sample")
    for axis in (axes[0, 0], axes[1, 0]):
        axis.axis("off")

    model_corr = correlation_map(model_spins)
    ref_corr = correlation_map(reference_spins)
    radius, model_g = radial_average_periodic(model_corr)
    _, ref_g = radial_average_periodic(ref_corr)
    valid = (radius > 0) & (model_g > 0) & (ref_g > 0)
    axes[0, 1].loglog(radius[valid], model_g[valid], label="model")
    axes[0, 1].loglog(radius[valid], ref_g[valid], label="MC")
    axes[0, 1].set_title("Connected G(r)")
    axes[0, 1].set_xlabel("r")
    axes[0, 1].legend()

    model_sf = structure_factor(model_spins)
    ref_sf = structure_factor(reference_spins)
    k_index, model_s = radial_average_periodic(model_sf)
    _, ref_s = radial_average_periodic(ref_sf)
    valid_k = (k_index > 0) & (model_s > 0) & (ref_s > 0)
    k = 2.0 * np.pi * k_index / model_spins.shape[-1]
    axes[0, 2].loglog(k[valid_k], model_s[valid_k], label="model")
    axes[0, 2].loglog(k[valid_k], ref_s[valid_k], label="MC")
    axes[0, 2].set_title("Structure factor S(k)")
    axes[0, 2].set_xlabel("|k|")
    axes[0, 2].legend()

    model_energy = -(
        model_spins * np.roll(model_spins, -1, -1)
        + model_spins * np.roll(model_spins, -1, -2)
    ).mean(axis=(-2, -1))
    ref_energy = -(
        reference_spins * np.roll(reference_spins, -1, -1)
        + reference_spins * np.roll(reference_spins, -1, -2)
    ).mean(axis=(-2, -1))
    axes[1, 1].hist(model_energy, bins=25, alpha=0.6, density=True, label="model")
    axes[1, 1].hist(ref_energy, bins=25, alpha=0.6, density=True, label="MC")
    axes[1, 1].set_title("Energy distribution")
    axes[1, 1].legend()

    model_mag = model_spins.mean(axis=(-2, -1))
    ref_mag = reference_spins.mean(axis=(-2, -1))
    axes[1, 2].hist(model_mag, bins=25, alpha=0.6, density=True, label="model")
    axes[1, 2].hist(ref_mag, bins=25, alpha=0.6, density=True, label="MC")
    axes[1, 2].set_title("Magnetization distribution")
    axes[1, 2].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = AxialDenoiser(DenoiserConfig(**config["model"])).to(device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    model.eval()
    diffusion = AbsorbingDiffusion(
        t_min=float(config["t_min"]),
        t_max=float(config.get("t_max", 1.0)),
        full_mask_probability=float(config.get("full_mask_probability", 0.05)),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated: list[np.ndarray] = []
    for start in range(0, args.samples, args.batch_size):
        batch = min(args.batch_size, args.samples - start)
        tokens = diffusion.sample(
            model,
            (batch, args.lattice_size, args.lattice_size),
            steps=args.steps,
            temperature=args.temperature,
            selection_noise=args.selection_noise,
            method=args.sampler,
            corrector_steps=args.corrector_steps,
            corrector_mask_ratio=args.corrector_mask_ratio,
            generator=generator,
        )
        generated.append((2 * tokens.cpu().numpy() - 1).astype(np.int8))
        print(f"Generated {start + batch}/{args.samples}")
    model_spins = np.concatenate(generated)
    reference_spins = load_reference(args)
    comparison = compare_ensembles(model_spins, reference_spins)
    comparison["checkpoint"] = str(args.checkpoint.resolve())
    comparison["sampling"] = {
        "steps": args.steps,
        "temperature": args.temperature,
        "sampler": args.sampler,
        "corrector_steps": args.corrector_steps,
        "corrector_mask_ratio": args.corrector_mask_ratio,
        "selection_noise": args.selection_noise,
        "seed": args.seed,
        "reference_split": args.reference_split,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        model=model_spins,
        reference=reference_spins,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    save_figure(args.output_dir / "comparison.png", model_spins, reference_spins)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
