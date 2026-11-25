"""Discrete masked diffusion for critical two-dimensional Ising configurations."""

from .ising import BETA_CRITICAL, WolffSampler

__all__ = [
    "AbsorbingDiffusion",
    "AxialDenoiser",
    "BETA_CRITICAL",
    "DenoiserConfig",
    "WolffSampler",
]


def __getattr__(name: str):
    """Load torch-dependent objects lazily so data diagnostics stay lightweight."""

    if name == "AbsorbingDiffusion":
        from .diffusion import AbsorbingDiffusion

        return AbsorbingDiffusion
    if name in {"AxialDenoiser", "DenoiserConfig"}:
        from .model import AxialDenoiser, DenoiserConfig

        return {
            "AxialDenoiser": AxialDenoiser,
            "DenoiserConfig": DenoiserConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
