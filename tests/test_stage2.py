from __future__ import annotations

import unittest

import numpy as np
import torch

from ism_diffusion.scale_data import coordinate_grid
from ism_diffusion.stage2_data import (
    sample_random_gap_windows,
    sample_random_pe_control_windows,
    stage2_geometry,
)
from ism_diffusion.stage2_model import (
    LocalGlobalDenoiserConfig,
    LocalGlobalScaleDenoiser,
)
from ism_diffusion.stage2_sampling import sample_with_method


def tiny_model(**overrides) -> LocalGlobalScaleDenoiser:
    options = dict(
        d_local=16,
        local_heads=2,
        local_blocks=1,
        d_global=32,
        global_heads=4,
        global_blocks=1,
        mlp_ratio=2.0,
        gate_hidden=16,
        physical_radius=1.0,
    )
    options.update(overrides)
    return LocalGlobalScaleDenoiser(LocalGlobalDenoiserConfig(**options))


class Stage2DataTests(unittest.TestCase):
    def test_variant_factorization(self):
        widths = [8, 12]
        spacings = [1, 2, 4]
        rng = np.random.default_rng(7)
        _, mode, _, coordinates = stage2_geometry(
            "LG-Gap-Matched", widths, spacings, rng
        )
        self.assertEqual(mode, "random_gap")
        self.assertEqual(coordinates, "matched")
        _, mode, _, coordinates = stage2_geometry(
            "LG-U-Unit", widths, spacings, rng
        )
        self.assertEqual(mode, "uniform")
        self.assertEqual(coordinates, "unit")
        _, mode, _, coordinates = stage2_geometry(
            "LG-U-RandPE", widths, spacings, rng
        )
        self.assertEqual(mode, "random_pe")
        self.assertEqual(coordinates, "randomized")

    def test_random_gap_coordinates_are_real_integer_offsets(self):
        parent = np.ones((2, 128, 128), dtype=np.int8)
        _, matched, _ = sample_random_gap_windows(
            parent,
            width=8,
            gaps=[1, 3, 5],
            coordinate_mode="matched",
            batch_size=3,
            rng=np.random.default_rng(11),
            device=torch.device("cpu"),
            augment=False,
        )
        dx = torch.diff(matched[:, :, 0, 0], dim=1)
        dy = torch.diff(matched[:, 0, :, 1], dim=1)
        allowed = torch.tensor([1.0, 3.0, 5.0])
        self.assertTrue(torch.isin(dx, allowed).all())
        self.assertTrue(torch.isin(dy, allowed).all())

        _, unit, _ = sample_random_gap_windows(
            parent,
            width=8,
            gaps=[1, 3, 5],
            coordinate_mode="unit",
            batch_size=2,
            rng=np.random.default_rng(11),
            device=torch.device("cpu"),
            augment=False,
        )
        self.assertTrue(torch.equal(torch.diff(unit[:, :, 0, 0], dim=1), torch.ones(2, 7)))
        self.assertTrue(torch.equal(torch.diff(unit[:, 0, :, 1], dim=1), torch.ones(2, 7)))

    def test_random_pe_matches_coordinate_distribution_not_spin_geometry(self):
        rng = np.random.default_rng(99)
        parents = rng.choice((-1, 1), size=(3, 256, 256)).astype(np.int8)
        gap_tokens, gap_coordinates, _ = sample_random_gap_windows(
            parents,
            width=12,
            gaps=[1, 2, 4, 8],
            coordinate_mode="matched",
            batch_size=4,
            rng=np.random.default_rng(123),
            device=torch.device("cpu"),
            augment=False,
        )
        pe_tokens, pe_coordinates, _ = sample_random_pe_control_windows(
            parents,
            width=12,
            gaps=[1, 2, 4, 8],
            batch_size=4,
            rng=np.random.default_rng(123),
            device=torch.device("cpu"),
            augment=False,
        )
        self.assertTrue(torch.equal(gap_coordinates, pe_coordinates))
        self.assertFalse(torch.equal(gap_tokens, pe_tokens))


class LocalGlobalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_forward_loss_shape_and_initial_gate(self):
        model = tiny_model().eval()
        tokens = torch.randint(0, 3, (2, 5, 5))
        coordinates = coordinate_grid(2, 5, 1, torch.device("cpu"))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        logits, diagnostic = model(
            tokens,
            torch.tensor([0.2, 0.8]),
            coordinates,
            valid,
            return_diagnostics=True,
        )
        self.assertEqual(logits.shape, (2, 2, 5, 5))
        self.assertTrue(torch.allclose(diagnostic["gate"], torch.full((2, 5, 5), 0.1)))
        self.assertTrue(torch.equal(diagnostic["global_residual"], torch.zeros_like(diagnostic["global_residual"])))

    def test_local_path_ignores_outside_radius_in_one_block(self):
        model = tiny_model().eval()
        torch.nn.init.normal_(model.local_output.weight)
        first = torch.zeros((1, 7, 7), dtype=torch.long)
        second = first.clone()
        second[0, 0, 0] = 1
        coordinates = coordinate_grid(1, 7, 1, torch.device("cpu"))
        valid = torch.ones_like(first, dtype=torch.bool)
        t = torch.tensor([0.5])
        _, a = model(first, t, coordinates, valid, return_diagnostics=True)
        _, b = model(second, t, coordinates, valid, return_diagnostics=True)
        self.assertTrue(
            torch.allclose(
                a["local_logits"][:, :, 3, 3],
                b["local_logits"][:, :, 3, 3],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_stacked_local_capacity_does_not_expand_spatial_radius(self):
        model = tiny_model(local_blocks=4).eval()
        torch.nn.init.normal_(model.local_output.weight)
        first = torch.zeros((1, 7, 7), dtype=torch.long)
        second = first.clone()
        second[0, 2, 2] = 1  # diagonal: Manhattan distance two from centre
        second[0, 1, 3] = 1  # two-hop axial site
        coordinates = coordinate_grid(1, 7, 1, torch.device("cpu"))
        valid = torch.ones_like(first, dtype=torch.bool)
        t = torch.tensor([0.5])
        _, a = model(first, t, coordinates, valid, return_diagnostics=True)
        _, b = model(second, t, coordinates, valid, return_diagnostics=True)
        self.assertTrue(
            torch.allclose(
                a["local_logits"][:, :, 3, 3],
                b["local_logits"][:, :, 3, 3],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_pad_frame_is_finite_and_invariant(self):
        model = tiny_model().eval()
        torch.nn.init.normal_(model.local_output.weight)
        tokens = torch.randint(0, 3, (1, 4, 4))
        coordinates = coordinate_grid(1, 4, 1, torch.device("cpu"))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        t = torch.tensor([0.6])
        small = model(tokens, t, coordinates, valid)

        padded_tokens = torch.full((1, 6, 6), 3, dtype=torch.long)
        padded_tokens[:, 1:5, 1:5] = tokens
        padded_coordinates = torch.zeros((1, 6, 6, 2))
        padded_coordinates[:, 1:5, 1:5] = coordinates
        padded_valid = torch.zeros((1, 6, 6), dtype=torch.bool)
        padded_valid[:, 1:5, 1:5] = True
        large = model(
            padded_tokens, t, padded_coordinates, padded_valid
        )[:, :, 1:5, 1:5]
        self.assertTrue(torch.isfinite(large).all())
        self.assertTrue(torch.allclose(small, large, atol=2e-6, rtol=2e-6))

    def test_hard_markov_gate_closes_with_four_visible_neighbours(self):
        model = tiny_model(hard_markov_gate=True).eval()
        tokens = torch.full((1, 5, 5), 2, dtype=torch.long)
        tokens[0, 1, 2] = 0
        tokens[0, 3, 2] = 1
        tokens[0, 2, 1] = 0
        tokens[0, 2, 3] = 1
        coordinates = coordinate_grid(1, 5, 1, torch.device("cpu"))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        _, diagnostic = model(
            tokens,
            torch.tensor([0.8]),
            coordinates,
            valid,
            return_diagnostics=True,
        )
        self.assertEqual(float(diagnostic["gate"][0, 2, 2]), 0.0)
        self.assertGreater(float(diagnostic["gate"][0, 0, 0]), 0.0)


class Stage2SamplerTests(unittest.TestCase):
    def test_all_sampler_candidates_finish(self):
        model = tiny_model().eval()
        coordinates = coordinate_grid(2, 4, 1, torch.device("cpu"))
        valid = torch.ones((2, 4, 4), dtype=torch.bool)
        for index, method in enumerate(("s0", "s1", "s2")):
            result = sample_with_method(
                model,
                coordinates,
                valid,
                method=method,
                steps=4,
                temperature=1.0,
                refinement_sweeps=1,
                generator=torch.Generator().manual_seed(20 + index),
            )
            self.assertTrue(torch.logical_or(result.eq(0), result.eq(1)).all())


if __name__ == "__main__":
    unittest.main()
