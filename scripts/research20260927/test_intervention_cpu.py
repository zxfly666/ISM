"""CPU-only regression checks for J setup/data; never a launch certificate.

This suite needs NumPy, not Torch, and reads the already verified local archive
for the exact I low-K blueprint. It never extracts or overwrites old artifacts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import tarfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import intervention_common as c
import intervention_data as d


def archived_blueprint():
    root = Path(c.CONFIG["base_backup_root"])
    if not root.exists():
        # On the authenticated Linux host verify the exact original science file;
        # local archive recovery remains a separate, already performed test.
        path = c.ROOT / "artifacts/geometry_identification_20260926/design/low_layouts.npz"
        digest = "dd82967bad896a785038892988b9a2fa6cfc9cd7327370cdfa570574cf5e1e55"
        if c.sha(path) != digest:
            raise RuntimeError("Remote I blueprint source mismatch")
        return c.load(path), dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest)
    manifest = c.read(root / "initial_source_protocol_preflight_v1.manifest.json")
    receipt = c.read(root / "initial_source_protocol_preflight_v1.verification.json")
    if receipt["status"] != "passed":
        raise RuntimeError("Unverified I archive")
    name = "artifacts/geometry_identification_20260926/design/low_layouts.npz"
    entry = next(row for row in manifest["files"] if row["path"] == name)
    archive = root / "initial_source_protocol_preflight_v1.tar.gz"
    if c.sha(archive) != receipt["archive_sha256"]:
        raise RuntimeError("I archive hash changed")
    with tarfile.open(archive, "r:gz") as tf:
        value = tf.extractfile(name).read()
    if len(value) != entry["bytes"] or hashlib.sha256(value).hexdigest() != entry["sha256"]:
        raise RuntimeError("I blueprint member mismatch")
    with np.load(io.BytesIO(value), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}, entry


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Software fixture, not an Ising chain or a new scientific reference.
        r = np.random.default_rng(98234019)
        cls.parent = SimpleNamespace(lattice_size=1024,
            spins=2*r.integers(0, 2, (2, 1024, 1024), dtype=np.int8)-1,
            chain_ids=np.array([0, 1]))
        cls.layouts, cls.blueprint_entry = archived_blueprint()

    def test_authorization_not_launch(self):
        self.assertTrue(c.verified_authorization()["authorized_to_start_once_only_if_all_gates_pass"])
        self.assertEqual(c.CONFIG["budget"]["hard_seconds"], 64800)
        self.assertEqual(c.CONFIG["status"], "design_only_not_implemented_not_preflighted_not_authorized_to_start")
        # Historical CONFIG is untouched; authorization is a separate artifact.

    def test_identities_and_block_order(self):
        cells = c.cells()
        self.assertEqual(len(cells), 24)
        self.assertEqual(len({x["cell"] for x in cells}), 24)
        count = {x["cell"]: 0 for x in cells}
        for ident, start, end in c.block_order():
            self.assertEqual(start, count[ident["cell"]])
            count[ident["cell"]] = end
        self.assertEqual(sum(count.values()), 336000)
        self.assertEqual(sorted(count.values()), [4000]*12+[24000]*12)
        fresh = {x["seed"] for x in cells if x["cohort"] == "fresh"}
        self.assertTrue(fresh.isdisjoint(c.CONFIG["base_seeds"]))

    def test_balanced_schedule_all_cycles(self):
        # Derive the frozen cycle permutation directly once/cycle (not 32 times).
        for ident in c.cells()[::2]:
            total_windows = total_sparse = 0
            for cycle in range(ident["updates"]//32):
                order = c.rng(ident["data_seed"], cycle, "balanced_schedule").permutation(32)
                self.assertEqual(sorted(order), list(range(32)))
                for value in order:
                    cell, repeat = divmod(int(value), 4)
                    total_windows += c.TOKENS // c.WIDTHS[cell % 4]**2
                    total_sparse += repeat == 3
            self.assertEqual(total_windows, 130000 if ident["cohort"] == "continuation" else 780000)
            self.assertEqual(total_sparse, ident["updates"]//4)

    def test_training_pairing_and_labels(self):
        for cohort in c.COHORTS:
            for index in range(6):
                left, right = [c.identity(cohort, index, mode) for mode in ("D", "O")]
                for step in list(range(1, 65))+[left["updates"]-1, left["updates"]]:
                    a = d.training_batch(self.parent, left, step)
                    b = d.training_batch(self.parent, right, step)
                    self.assertEqual(a["paired_data_hash"], b["paired_data_hash"])
                    self.assertEqual(a["actual_input_hash"], b["actual_input_hash"])
                    self.assertEqual(a["clean"].size, 18432)
                    self.assertTrue(np.isfinite(a["loss_weights"]).all())
                    if a["sparse"]:
                        self.assertEqual(int(a["query_valid"].sum()), 512)
                        rows = np.arange(a["batch"])[:, None]
                        flat = a["noisy"].reshape(a["batch"], -1)
                        self.assertTrue(np.all(flat[rows, a["queries"]][a["query_valid"]] == 2))
                        y = a["clean"].reshape(a["batch"], -1)[rows, a["queries"]]
                        np.testing.assert_array_equal(y[a["query_valid"]], a["labels"][a["query_valid"]])
                        np.testing.assert_allclose(a["loss_weights"].sum(), 1.)

    def test_core_spans_nested_evidence(self):
        for width, span in ((48, 174), (96, 354)):
            b = d.bank(self.parent, np.arange(2), 2026092721, width, width, "held_gap", 512)
            d.validate_bank(b)
            self.assertTrue(np.all(b["physical_axes"][:, :, -1] == span))
            self.assertEqual(set(np.unique(np.diff(b["physical_axes"], axis=-1))), {3, 6})
        a, b = [d.bank(self.parent, np.arange(2), 2026092721, 3, 48, "continuous", k) for k in (115, 1152)]
        np.testing.assert_array_equal(a["queries"], b["queries"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        np.testing.assert_array_equal(a["evidence"], b["evidence"][:, :115])

    def test_factorial_physical_identity(self):
        banks = d.mechanism_banks(self.parent, np.arange(2))
        self.assertEqual(len(banks), 8)
        baseline = banks["N576_D48_t576"]
        for name, b in banks.items():
            d.validate_bank(b)
            np.testing.assert_array_equal(b["labels"], baseline["labels"])
            np.testing.assert_array_equal(b["input_coordinates"][:, :, :96], baseline["input_coordinates"][:, :, :96])
            np.testing.assert_array_equal(b["noisy"][:, :, :96], baseline["noisy"][:, :, :96])
            domain = int(b["domain"])
            lo = 24 if domain == 48 else 0
            self.assertEqual(float(b["input_coordinates"].min()), lo)
            self.assertEqual(float(b["input_coordinates"].max()), lo+domain-1)
            for coords in b["input_coordinates"][:, 0]:
                self.assertEqual(len(np.unique(coords, axis=0)), int(b["token_count"]))
        for domain in (48, 96):
            np.testing.assert_array_equal(banks[f"N576_D{domain}_t576"]["input_coordinates"],
                                          banks[f"N2304_D{domain}_t576"]["input_coordinates"][:, :, :576])

    def test_padding_ordinary_mask_not_pad(self):
        for k in (32, 512):
            a, b, z = d.padding_banks(self.parent, np.arange(2), k).values()
            for bank in (a, b, z):
                d.validate_bank(bank)
                self.assertNotIn(3, np.unique(bank["noisy"]))
            np.testing.assert_array_equal(a["noisy"], b["noisy"][:, 24:72, 24:72])
            np.testing.assert_array_equal(a["input_coordinates"], b["input_coordinates"][:, 24:72, 24:72])
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_array_equal(a["t"], b["t"])
            self.assertTrue(np.all(z["t"] > b["t"]))

    def test_exact_gibbs_soft_targets(self):
        signs, target = d.gibbs_patterns()
        beta = np.log(1+np.sqrt(2))/2
        # Independent two-state enumeration using local Hamiltonian weights.
        weights = np.exp(beta*signs.sum(1)[:, None]*np.array([-1, 1]))
        np.testing.assert_allclose(target, weights[:, 1]/weights.sum(1), atol=1e-15)
        np.testing.assert_allclose(target+target[::-1], 1.)
        d.validate_bank(d.fixture_bank())
        banks = d.oracle_banks(self.parent, np.arange(2))
        self.assertEqual(len(banks), 9)
        for name, b in banks.items():
            d.validate_bank(b)
            np.testing.assert_array_equal(b["labels"].reshape(2, 16), np.tile(target, (2, 1)))
            if int(b["k"]) == 4:
                # Duplicate K4 backgrounds are not independent capability data.
                np.testing.assert_array_equal(b["noisy"][:16], b["noisy"][16:])
            if int(b["width"]) == 96:
                self.assertTrue(np.all(b["queries"] == 47*96+47))

    def test_low_blueprint_and_common_translation(self):
        b = d.low_inputs(self.layouts)
        self.assertEqual(b["noisy"].shape, (960, 48, 48))
        self.assertTrue(np.isnan(b["labels"]).all())
        self.assertEqual(str(b["target_type"]), "reference_only")
        r = d.low_reference(self.parent, self.layouts)
        self.assertEqual(r["counts"].shape, (2, 80, 2, 8))
        self.assertTrue(np.all(r["counts"].sum(-1) == 256))
        self.assertEqual(self.blueprint_entry["path"], "artifacts/geometry_identification_20260926/design/low_layouts.npz")

    def test_no_old_globals_imported(self):
        for file in Path(__file__).parent.glob("intervention_*.py"):
            tree = ast.parse(file.read_text("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("identification_"))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not x.name.startswith("identification_") for x in node.names))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", type=Path)
    args = p.parse_args()
    if args.receipt:
        allowed = (c.ROOT / "artifacts/observed_context_intervention_20260927_preparation").resolve()
        if not args.receipt.resolve().is_relative_to(allowed) or args.receipt.exists():
            raise ValueError("Use a new receipt in the J local preparation directory")
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataTests))
    record = dict(status="passed_CPU_data_only" if result.wasSuccessful() else "failed_CPU_data_tests",
        time=time.time(), seconds=time.time()-started, tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        files={x.relative_to(c.ROOT).as_posix(): c.sha(x) for x in Path(__file__).parent.glob("intervention_*.py")},
        authorization_sha256=c.sha(c.AUTH_PATH), config_sha256=c.sha(c.CONFIG_PATH),
        not_performed=["Torch_model_equivalence", "Torch_checkpoint_recovery", "fixed_1024_step_ability_fixture",
                       "GPU_preflight", "MC", "formal_training", "full_scratch_pipeline", "launch_budget_gate"],
        launch_permitted_by_this_receipt=False, budget_started=None)
    if args.receipt:
        c.write(args.receipt, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
