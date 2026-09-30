"""Torch/CPU software checks; run separately from the charged GPU preflight.

These nonzero-head synthetic checks do not perform the official 1024-update
Gibbs capability fixture and cannot certify scientific learnability.
"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
import intervention_common as c
import intervention_model as im


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(481092)
        self.old = CoordinateDenseDenoiser(CoordinateDenoiserConfig(**c.MODEL)).eval()
        # Exercise actual attention dependence, not the trivially constant zero head.
        with torch.no_grad():
            self.old.output.weight.normal_(0, .04)
            self.old.output.bias.normal_(0, .02)
        self.dense = im.InterventionDenoiser(self.old.config, "dense").eval()
        self.observed = im.InterventionDenoiser(self.old.config, "observed_only").eval()
        self.dense.load_state_dict(self.old.state_dict())
        self.observed.load_state_dict(self.old.state_dict())
        self.x = torch.full((2, 8, 8), 2, dtype=torch.long)
        self.x[:, 1, 2] = torch.tensor([0, 1])
        self.x[:, 5, 4] = torch.tensor([1, 0])
        axis = torch.arange(8, dtype=torch.float32)
        self.xy = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1)[None].repeat(2, 1, 1, 1)
        self.t = torch.full((2,), 1-2/64)
        self.v = torch.ones_like(self.x, dtype=torch.bool)

    def prob(self, model, x=None, xy=None, valid=None):
        with torch.no_grad():
            return model(self.x if x is None else x, self.t,
                         self.xy if xy is None else xy,
                         self.v if valid is None else valid).softmax(1)[:, 1].flatten(1)

    def assert_close(self, a, b):
        self.assertTrue(bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        self.assertLessEqual(float((a-b).abs().max()), 2e-5)

    def test_same_parameter_registration_and_dense_forward(self):
        old = list(self.old.named_parameters())
        new = list(self.dense.named_parameters())
        self.assertEqual([(k, tuple(v.shape)) for k, v in old], [(k, tuple(v.shape)) for k, v in new])
        self.assertEqual(sum(p.numel() for _, p in new), 1976706)
        self.assertGreater(float(self.prob(self.old).std()), 1e-4)
        self.assertTrue(torch.equal(self.prob(self.old), self.prob(self.dense)))

    def test_every_layer_uses_separate_key_mask(self):
        original = im.attention
        masks = []
        def capture(attn, x, xy, valid, keys):
            masks.append((valid.clone(), keys.clone()))
            return original(attn, x, xy, valid, keys)
        with patch.object(im, "attention", capture):
            self.prob(self.observed)
        self.assertEqual(len(masks), 7)
        for valid, keys in masks:
            self.assertTrue(bool(valid.all()))
            self.assertTrue(bool((keys.sum(1) == 2).all()))

    def test_batch_and_mixed_zero_observation_fallback(self):
        x = self.x.clone(); x[0] = 2
        keys = im.allowed_keys(x, self.v, "observed_only")
        self.assertTrue(bool(keys[0].all()))
        self.assertEqual(int(keys[1].sum()), 2)
        with torch.no_grad():
            batch = self.observed(x, self.t, self.xy).softmax(1)
            each = torch.cat([self.observed(x[i:i+1], self.t[i:i+1], self.xy[i:i+1]).softmax(1) for i in range(2)])
            dense0 = self.dense(x[:1], self.t[:1], self.xy[:1]).softmax(1)
        self.assert_close(batch, each)
        self.assert_close(batch[:1], dense0)

    def test_flatten_permutation_and_translation(self):
        permutation = torch.randperm(64)
        inverse = torch.argsort(permutation)
        for model in (self.dense, self.observed):
            p = self.prob(model)
            self.assert_close(p, self.prob(model, self.x.reshape(2, 1, 64), self.xy.reshape(2, 1, 64, 2), self.v.reshape(2, 1, 64)))
            px = self.x.flatten(1)[:, permutation].reshape(2, 1, 64)
            xy = self.xy.reshape(2, 64, 2)[:, permutation].reshape(2, 1, 64, 2)
            v = self.v.flatten(1)[:, permutation].reshape(2, 1, 64)
            self.assert_close(p, self.prob(model, px, xy, v)[:, inverse])
            for shift in ((7, -11), (-13, 5)):
                self.assert_close(p, self.prob(model, xy=self.xy+torch.tensor(shift)))

    def test_pad_equivalence_is_not_mask_equivalence(self):
        flat_x = self.x.reshape(2, 1, 64)
        flat_xy = self.xy.reshape(2, 1, 64, 2)
        extra_xy = torch.arange(32, dtype=torch.float32).reshape(1, 1, 16, 2).repeat(2, 1, 1, 1)+10
        xy = torch.cat([flat_xy, extra_xy], 2)
        valid = torch.cat([self.v.reshape(2, 1, 64), torch.zeros((2, 1, 16), dtype=torch.bool)], 2)
        padded = torch.cat([flat_x, torch.full((2, 1, 16), 3)], 2)
        for model in (self.dense, self.observed):
            self.assert_close(self.prob(model), self.prob(model, padded, xy, valid)[:, :64])
        masked = torch.cat([flat_x, torch.full((2, 1, 16), 2)], 2)
        self.assert_close(self.prob(self.observed), self.prob(self.observed, masked, xy, torch.ones_like(masked, dtype=torch.bool))[:, :64])
        # Ensure this test has sensitivity to the background path of D.
        delta = (self.prob(self.dense)-self.prob(self.dense, masked, xy, torch.ones_like(masked, dtype=torch.bool))[:, :64]).abs().max()
        self.assertGreater(float(delta), 2e-5)

    def test_valid_hidden_queries_still_receive_gradients(self):
        for model in (self.dense, self.observed):
            model.zero_grad(set_to_none=True)
            logits = model(self.x, self.t, self.xy)
            loss = -logits[:, :, 3, 3].log_softmax(1)[:, 1].mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertGreater(float(model.output.weight.grad.norm()), 0.)
            self.assertGreater(float(model.token_embedding.weight.grad[2].norm()), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
