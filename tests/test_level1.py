from __future__ import annotations

import unittest

import numpy as np
import torch

from ism_diffusion.scale_data import (
    coordinate_grid,
    pack_spins,
    unpack_spins,
    variant_geometry,
)
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.scale_evaluation import (
    crop_periodic_windows,
    open_energy_density,
    open_radial_correlation,
)
from ism_diffusion.scale_model import (
    CoordinateDenseDenoiser,
    CoordinateDenoiserConfig,
)


class ScaleDataTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        rng = np.random.default_rng(3)
        spins = rng.choice((-1, 1), size=(5, 17, 17)).astype(np.int8)
        restored = unpack_spins(pack_spins(spins), 17)
        self.assertTrue(np.array_equal(spins, restored))

    def test_variants_have_expected_strides(self):
        widths = [16, 24, 32, 48]
        strides = [1, 2, 4]
        self.assertEqual(
            variant_geometry("T0", widths, strides, np.random.default_rng(1)),
            (48, 1, 1),
        )
        for variant in ("T3", "Pphase", "Punit"):
            width, spin_stride, coordinate_stride = variant_geometry(
                variant, widths, strides, np.random.default_rng(4)
            )
            self.assertIn(width, widths)
            if variant == "T3":
                self.assertEqual(spin_stride, coordinate_stride)
            elif variant == "Pphase":
                self.assertEqual(spin_stride, 1)
            else:
                self.assertEqual(coordinate_stride, 1)


class CoordinateModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = CoordinateDenseDenoiser(
            CoordinateDenoiserConfig(
                d_model=32,
                n_heads=4,
                n_blocks=2,
                mlp_ratio=2.0,
            )
        ).eval()
        self.diffusion = CoordinateAbsorbingDiffusion(t_min=0.05)

    def test_forward_and_loss(self):
        clean = torch.randint(0, 2, (2, 4, 4))
        coordinates = coordinate_grid(2, 4, 3, torch.device("cpu"))
        valid = torch.ones_like(clean, dtype=torch.bool)
        logits = self.model(clean, torch.full((2,), 0.5), coordinates, valid)
        self.assertEqual(logits.shape, (2, 2, 4, 4))
        result = self.diffusion.training_loss(
            self.model, clean, coordinates, valid
        )
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()

    def test_global_coordinate_translation_invariance(self):
        tokens = torch.randint(0, 3, (2, 4, 4))
        coordinates = coordinate_grid(2, 4, 2, torch.device("cpu"))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        t = torch.tensor([0.3, 0.8])
        first = self.model(tokens, t, coordinates, valid)
        shifted = self.model(tokens, t, coordinates + 37.25, valid)
        self.assertTrue(torch.allclose(first, shifted, atol=2e-6, rtol=2e-6))

    def test_pad_frame_invariance(self):
        tokens = torch.randint(0, 3, (1, 4, 4))
        coordinates = coordinate_grid(1, 4, 1, torch.device("cpu"))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        t = torch.tensor([0.6])
        small = self.model(tokens, t, coordinates, valid)

        padded_tokens = torch.full((1, 6, 6), 3, dtype=torch.long)
        padded_tokens[:, 1:5, 1:5] = tokens
        padded_coordinates = torch.zeros((1, 6, 6, 2))
        padded_coordinates[:, 1:5, 1:5] = coordinates
        padded_valid = torch.zeros((1, 6, 6), dtype=torch.bool)
        padded_valid[:, 1:5, 1:5] = True
        large = self.model(
            padded_tokens, t, padded_coordinates, padded_valid
        )[:, :, 1:5, 1:5]
        self.assertTrue(torch.allclose(small, large, atol=2e-6, rtol=2e-6))

    def test_coordinate_sampler_finishes(self):
        coordinates = coordinate_grid(2, 4, 1, torch.device("cpu"))
        valid = torch.ones((2, 4, 4), dtype=torch.bool)
        samples = self.diffusion.sample(
            self.model,
            coordinates,
            valid,
            steps=3,
            generator=torch.Generator().manual_seed(8),
        )
        self.assertTrue(torch.logical_or(samples.eq(0), samples.eq(1)).all())


class ScaleEvaluationTests(unittest.TestCase):
    def test_open_correlation_has_no_wraparound_bias_for_constant_field(self):
        spins = np.ones((3, 8, 8), dtype=np.int8)
        radii, raw, connected = open_radial_correlation(spins, max_radius=4)
        self.assertTrue(np.array_equal(radii, np.arange(5)))
        self.assertTrue(np.allclose(raw, 1.0, atol=1e-12))
        self.assertTrue(np.allclose(connected, 0.0, atol=1e-12))

    def test_open_energy_uses_only_internal_bonds(self):
        row, column = np.indices((6, 6))
        checkerboard = (1 - 2 * ((row + column) % 2)).astype(np.int8)
        energy = open_energy_density(checkerboard[None])
        self.assertTrue(np.allclose(energy, 1.0))

    def test_periodic_crop_respects_physical_stride(self):
        parent = np.arange(64, dtype=np.int8).reshape(1, 8, 8)
        crop = crop_periodic_windows(
            parent,
            np.asarray([0]),
            np.asarray([7]),
            np.asarray([7]),
            width=3,
            spin_stride=2,
        )
        expected_indices = np.asarray([7, 1, 3])
        expected = parent[0][np.ix_(expected_indices, expected_indices)]
        self.assertTrue(np.array_equal(crop[0], expected))


if __name__ == "__main__":
    unittest.main()
