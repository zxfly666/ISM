from __future__ import annotations

import unittest

import numpy as np
import torch

from ism_diffusion.diffusion import AbsorbingDiffusion
from ism_diffusion.ising import WolffSampler, energy_density
from ism_diffusion.metrics import ensemble_summary
from ism_diffusion.model import AxialDenoiser, DenoiserConfig
from train import augment


class WolffTests(unittest.TestCase):
    def test_sampler_returns_binary_periodic_lattices(self):
        sampler = WolffSampler(lattice_size=6, seed=3)
        sampler.thermalize(2)
        samples = sampler.sample(4, sweeps_between=1)
        self.assertEqual(samples.shape, (4, 6, 6))
        self.assertTrue(np.isin(samples, (-1, 1)).all())
        self.assertTrue(np.isfinite(energy_density(samples)).all())

    def test_ensemble_metrics_are_finite(self):
        rng = np.random.default_rng(1)
        samples = rng.choice((-1, 1), size=(16, 8, 8)).astype(np.int8)
        summary = ensemble_summary(samples)
        for key in ("energy_mean", "binder_u4", "xi_over_l"):
            self.assertTrue(np.isfinite(summary[key]))


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = AxialDenoiser(
            DenoiserConfig(
                d_model=16,
                n_heads=4,
                n_blocks=1,
                mlp_ratio=2.0,
            )
        )
        self.diffusion = AbsorbingDiffusion(
            t_min=0.05, full_mask_probability=0.1
        )

    def test_endpoint_corruption_is_all_mask(self):
        clean = torch.randint(0, 2, (2, 4, 4))
        noisy, masked = self.diffusion.corrupt(clean, torch.ones(2))
        self.assertTrue(masked.all())
        self.assertTrue(noisy.eq(2).all())

    def test_loss_is_finite_and_backpropagates(self):
        clean = torch.randint(0, 2, (2, 4, 4))
        result = self.diffusion.training_loss(self.model, clean)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_sampler_finishes_without_mask_tokens(self):
        generator = torch.Generator().manual_seed(9)
        sample = self.diffusion.sample(
            self.model, (2, 4, 4), steps=3, generator=generator
        )
        self.assertEqual(sample.shape, (2, 4, 4))
        self.assertTrue(torch.logical_or(sample.eq(0), sample.eq(1)).all())

    def test_sampler_preserves_known_region(self):
        shape = (2, 4, 4)
        known_tokens = torch.randint(0, 2, shape)
        known_mask = torch.zeros(shape, dtype=torch.bool)
        known_mask[:, :2, :2] = True
        sample = self.diffusion.sample(
            self.model,
            shape,
            steps=3,
            corrector_steps=2,
            corrector_mask_ratio=0.5,
            known_tokens=known_tokens,
            known_mask=known_mask,
            generator=torch.Generator().manual_seed(10),
        )
        self.assertTrue(torch.equal(sample[known_mask], known_tokens[known_mask]))

    def test_confidence_sampler_remains_available(self):
        sample = self.diffusion.sample(
            self.model,
            (1, 4, 4),
            steps=3,
            method="confidence",
            generator=torch.Generator().manual_seed(11),
        )
        self.assertTrue(torch.logical_or(sample.eq(0), sample.eq(1)).all())

    def test_confidence_corrector_finishes(self):
        sample = self.diffusion.sample(
            self.model,
            (1, 4, 4),
            steps=4,
            method="confidence_corrector",
            corrector_steps=2,
            corrector_mask_ratio=0.2,
            generator=torch.Generator().manual_seed(12),
        )
        self.assertTrue(torch.logical_or(sample.eq(0), sample.eq(1)).all())


class AugmentationTests(unittest.TestCase):
    def test_d4_and_spin_flip_preserve_ising_energy(self):
        tokens = torch.randint(0, 2, (32, 8, 8))
        spins = 2 * tokens - 1
        original = -(
            spins * torch.roll(spins, -1, dims=-1)
            + spins * torch.roll(spins, -1, dims=-2)
        ).sum(dim=(-2, -1))
        augmented = augment(tokens, torch.Generator().manual_seed(123))
        augmented_spins = 2 * augmented - 1
        transformed = -(
            augmented_spins * torch.roll(augmented_spins, -1, dims=-1)
            + augmented_spins * torch.roll(augmented_spins, -1, dims=-2)
        ).sum(dim=(-2, -1))
        self.assertTrue(torch.equal(original, transformed))
        self.assertTrue(torch.logical_or(augmented.eq(0), augmented.eq(1)).all())


if __name__ == "__main__":
    unittest.main()
