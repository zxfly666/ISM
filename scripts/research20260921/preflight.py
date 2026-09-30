"""Finite S0 checks and actual-device benchmark; does not train a study arm."""
import copy
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ism_diffusion.geometry_study import (
    GAPS, WIDTHS, TOKENS_PER_UPDATE, atomic_json, array_hash, file_hash,
    make_batch, corrupt_batch, coordinate_arrays, new_model, train_update,
)
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model, centered_coordinate_grid
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from ism_diffusion.ising import BETA_CRITICAL


OUT = ROOT / "artifacts/geometry_alignment_20260921/preflight"


@torch.inference_mode()
def probability(model, tokens, coords, valid, t):
    result = []
    for start in range(0, len(tokens), 2):
        sl = slice(start, start + 2)
        logits = model(tokens[sl], torch.full((len(tokens[sl]),), t, device="cuda"),
                       coords[sl], valid[sl])
        result.append(logits.float().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def s0():
    patterns = np.asarray(list(itertools.product((-1, 1), repeat=4)))
    exact = 1 / (1 + np.exp(-2 * BETA_CRITICAL * patterns.sum(1)))
    paths = {"Dense": "stage2a/Dense_T3_plus", "LocalGlobal": "stage2a/LG_T3"}
    report = dict(gate_i={}, physics_diagnostics={}, checkpoints={},
                  warning="16-pattern stress is exhaustive, not a natural sample distribution; no state bootstrap.")
    saved = {"patterns": patterns, "exact_p_plus": exact}
    for name, rel in paths.items():
        path = ROOT / "results/scale_aware_context/checkpoints" / rel / "best_val.pt"
        model, payload = load_scale_model(path, torch.device("cuda"))
        report["checkpoints"][name] = dict(sha256=file_hash(path), step=payload["step"], state="ema")
        model_values = {}
        for width in (3, 16, 64):
            c = width // 2
            coords = centered_coordinate_grid(16, width, 1., torch.device("cuda"))
            for context in (("pad",) if width == 3 else ("pad", "mask")):
                tokens = torch.full((16, width, width), 3 if context == "pad" else 2,
                                    dtype=torch.long, device="cuda")
                valid = torch.zeros_like(tokens, dtype=torch.bool) if context == "pad" else torch.ones_like(tokens, dtype=torch.bool)
                tokens[:, c, c] = 2
                valid[:, c, c] = True
                for j, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    tokens[:, c+dx, c+dy] = torch.tensor((patterns[:, j]+1)//2, device="cuda")
                    valid[:, c+dx, c+dy] = True
                for tv in (.2, .5, .8, .95):
                    pp = probability(model, tokens, coords, valid, tv)[:, c, c]
                    key = f"w{width}_{context}_t{tv}"
                    model_values[key] = pp
                    saved[f"{name}_{key}"] = pp
                if width == 16 and context == "mask":
                    base = probability(model, tokens[:2], coords[:2], valid[:2], .5)
                    moved = probability(model, tokens[:2], coords[:2] + torch.tensor([11., -7.], device="cuda"), valid[:2], .5)
                    permutation = torch.randperm(width**2, generator=torch.Generator().manual_seed(10)).to("cuda")
                    perm_tokens = tokens[:2].flatten(1)[:, permutation].reshape(2, width, width)
                    perm_coords = coords[:2].reshape(2, -1, 2)[:, permutation].reshape(2, width, width, 2)
                    perm_valid = valid[:2].flatten(1)[:, permutation].reshape(2, width, width)
                    perm_probs = probability(model, perm_tokens, perm_coords, perm_valid, .5)
                    back = perm_probs.reshape(2, -1)[:, permutation.argsort().cpu().numpy()].reshape(base.shape)
                    report["gate_i"][f"{name}_translation"] = float(np.max(np.abs(base-moved)))
                    report["gate_i"][f"{name}_permutation"] = float(np.max(np.abs(base-back)))
        pad_error = max(float(np.max(np.abs(model_values[f"w3_pad_t{tv}"]-model_values[f"w{w}_pad_t{tv}"])))
                        for tv in (.2, .5, .8, .95) for w in (16, 64))
        report["gate_i"][f"{name}_pad"] = pad_error
        report["physics_diagnostics"][name] = {}
        for key, pp in model_values.items():
            q = np.clip(pp, 1e-7, 1-1e-7)
            kl = exact*np.log(exact/q)+(1-exact)*np.log((1-exact)/(1-q))
            report["physics_diagnostics"][name][key] = dict(mean_kl=float(kl.mean()),
                mean_abs_error=float(np.abs(pp-exact).mean()), max_abs_error=float(np.abs(pp-exact).max()))
        print(json.dumps(dict(s0_model=name, gate_i=report["gate_i"])), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    report["gate_i_pass"] = all(np.isfinite(v) and v < 5e-5 for v in report["gate_i"].values())
    np.savez_compressed(OUT / "s0_arrays.npz", **saved)
    atomic_json(OUT / "s0.json", report)
    if not report["gate_i_pass"]:
        raise RuntimeError("Structural invariant failure: inspect before main training")


def data_tests(parent):
    results = {}
    for width in WIDTHS:
        for kind in ("continuous", "gap"):
            one = make_batch(parent, 4, width, kind, GAPS, 817, width, True)
            two = make_batch(parent, 4, width, kind, GAPS, 817, width, True)
            assert array_hash(one["clean"]) == array_hash(two["clean"])
            for arm in ("A", "B", "C"):
                assert np.array_equal(one["coords"][arm], two["coords"][arm])
            assert np.array_equal(corrupt_batch(one["clean"], 817, width)[1],
                                  corrupt_batch(two["clean"], 817, width)[1])
            if kind == "continuous":
                assert all(np.array_equal(one["coords"]["A"], one["coords"][a]) for a in ("B", "C"))
            else:
                assert not np.array_equal(one["coords"]["A"], one["coords"]["C"])
            results[f"{kind}/{width}"] = "pass"
    # Actual sparse sampling, before augmentation: compare all tokens to their
    # addressed parent sites, including cases that wrap around the parent.
    b = make_batch(parent, 8, 48, "gap", GAPS, 92, 11, False)
    for row in range(8):
        x = (b["origin"][row, 0] + b["axes"][0][row]) % parent.lattice_size
        y = (b["origin"][row, 1] + b["axes"][1][row]) % parent.lattice_size
        expected = parent.spins[b["parent"][row]][np.ix_(x, y)] > 0
        assert np.array_equal(expected, b["clean"][row])
    results["physical_indexing"] = "pass"
    return results


def benchmark(parent):
    model = new_model(9823)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.05, fused=True)
    times, microbatches = {}, {}
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for width in WIDTHS:
        total_batch = TOKENS_PER_UPDATE // width**2
        micro = total_batch
        while True:
            try:
                batch = make_batch(parent, total_batch, width, "gap", GAPS, 9823, 901, True)
                train_update(model, ema, optimizer, batch, "A", 9823, 1, 16000, micro)
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                micro //= 2
                if micro < 1:
                    raise
        microbatches[str(width)] = micro
        for kind in ("continuous", "gap"):
            elapsed = []
            for repeat in range(12):
                step += 1
                torch.cuda.synchronize()
                start = time.perf_counter()
                batch = make_batch(parent, total_batch, width, kind, GAPS, 9823, step, True)
                train_update(model, ema, optimizer, batch, "A", 9823, step, 16000, micro)
                torch.cuda.synchronize()
                if repeat >= 3:
                    elapsed.append(time.perf_counter()-start)
            times[f"{kind}/w{width}"] = dict(median=float(np.median(elapsed)), max=float(max(elapsed)))
            print(json.dumps(dict(benchmark=f"{kind}/w{width}", **times[f"{kind}/w{width}"])), flush=True)
    training_peak = torch.cuda.max_memory_allocated()/2**30
    diffusion = CoordinateAbsorbingDiffusion()
    sample_times = {}
    model.eval()
    for width in (32, 48, 64, 96):
        coords, _ = coordinate_arrays(4, width, "continuous", (1,), 53, width)
        cc = torch.tensor(coords["A"], device="cuda")
        valid = torch.ones((4, width, width), dtype=torch.bool, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            diffusion.sample(model, cc, valid, steps=32,
                             generator=torch.Generator(device="cuda").manual_seed(817))
        torch.cuda.synchronize()
        sample_times[str(width)] = (time.perf_counter()-start)/4*8
    result = dict(training_seconds_per_update=float(np.mean([x["median"] for x in times.values()])),
                  cells=times, microbatch=microbatches, sample_seconds_per_256step=sample_times,
                  training_peak_allocated_gib=training_peak,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__)
    atomic_json(OUT / "benchmark.json", result)
    print(json.dumps(result), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    path = ROOT / "data/level1/parents_l1024.npz"
    digest = file_hash(path)
    if digest != "1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934":
        raise RuntimeError("Parent dataset hash mismatch")
    s0()
    parent = load_parent_split(path, "train")
    tests = data_tests(parent)
    atomic_json(OUT / "data_contract.json", dict(dataset_sha256=digest, tests=tests, metadata=parent.metadata))
    torch.set_float32_matmul_precision("high")
    benchmark(parent)
    atomic_json(OUT / "complete.json", dict(status="preflight_complete", time=time.time()))


if __name__ == "__main__":
    main()
