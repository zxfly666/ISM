"""Fixed-NFE reveal and committed-spin repair, with paired per-image RNG."""
from __future__ import annotations

import math
import numpy as np
import torch
import endpoint_common as c
import endpoint_data as d


def image_seeds(seed, width, ids, role):
    root = c.CFG["role_seeds"]["repair" if role == "repair" else "generation"]
    return np.array([c.stream(root, int(i), f"s{seed}_w{width}_{role}") for i in ids], dtype=np.int64)


def repair_design(seed, width, ids, steps=64):
    count = 32 if width == 96 else 8 if width == 48 else min(2, len(d.checkerboard_positions(width, 0)))
    positions = np.empty((steps, len(ids), count), dtype=np.int64)
    uniforms = np.empty_like(positions, dtype=np.float32)
    seeds = image_seeds(seed, width, ids, "repair")
    for i, s in enumerate(seeds):
        r = np.random.default_rng(int(s))
        for step in range(steps):
            positions[step, i] = r.choice(d.checkerboard_positions(width, step % 2), size=count, replace=False)
            uniforms[step, i] = r.random(count).astype(np.float32)
    return positions, uniforms, seeds


@torch.inference_mode()
def sample(model, seed, width, ids, sampler="monotone-256", clock=.002, amp=True, check=c.check, toy=False):
    assert sampler in ("monotone-256", "reveal192-repair64")
    assert width in (48, 96) or toy
    model.eval()
    device = next(model.parameters()).device
    n = len(ids)
    reveal_steps = 256 if sampler == "monotone-256" else 192
    repairs = 0 if sampler == "monotone-256" else 64
    if toy:
        reveal_steps = 8 if repairs == 0 else 6
        repairs = 0 if repairs == 0 else 2
    seeds = image_seeds(seed, width, ids, "reveal")
    # Same maximum stream shape for both samplers and floors. Per-image independent.
    total_steps = 8 if toy else 256
    u = torch.stack([torch.rand((total_steps, 2, width, width), device=device,
                               generator=torch.Generator(device=device).manual_seed(int(s))) for s in seeds], 1)
    xy = torch.as_tensor(d.coordinates(np.broadcast_to(np.arange(width), (n, 2, width)).copy()), device=device)
    valid = torch.ones((n, width, width), device=device, dtype=torch.bool)
    tokens = torch.full((n, width, width), 2, device=device, dtype=torch.long)
    times = np.cos(.5*np.pi*np.arange(reveal_steps+1)/reveal_steps)**2
    times[-1] = 0.
    trace_count, trace_t, trace_reveal = [], [], []
    calls, active_calls = 0, torch.zeros(n, dtype=torch.int64, device=device)
    for j in range(reveal_steps):
        check()
        mask = tokens == 2
        count = mask.sum((1, 2))
        active_calls += count > 0
        t, s = times[j:j+2]
        mt = torch.full((n,), max(float(t), clock), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(tokens, mt, xy, valid)
        prob = logits.float().softmax(1)[:, 1]
        new = (u[j, :, 0] < prob).long()
        reveal = mask & (u[j, :, 1] < float(min(max(1-s/max(t, 1e-12), 0), 1)))
        tokens = torch.where(reveal, new, tokens)
        trace_count.append(count); trace_t.append(mt); trace_reveal.append(reveal.sum((1, 2)))
        calls += 1
    assert not bool((tokens == 2).any())
    result = dict(image_ids=np.asarray(ids), reveal_rng_seeds=seeds,
                  nominal_t=times[:-1], model_t=torch.stack(trace_t).T.cpu().numpy(),
                  M_before=torch.stack(trace_count).T.cpu().numpy(), newly_revealed=torch.stack(trace_reveal).T.cpu().numpy(),
                  active_reveal_calls=active_calls.cpu().numpy(), sampler=np.array(sampler), clock_floor=np.array(clock))
    if repairs:
        prefix = tokens.clone()
        oracle = tokens.clone()
        positions, uniforms, repair_seeds = repair_design(seed, width, ids, repairs)
        pidx = torch.as_tensor(positions, device=device)
        pu = torch.as_tensor(uniforms, device=device)
        rows = torch.arange(n, device=device)[:, None]
        prob_trace, target_trace, oracle_target_trace = [], [], []
        old_trace, new_trace, oracle_old_trace, oracle_new_trace = [], [], [], []
        neighbor_trace, oracle_neighbor_trace = [], []
        for j in range(repairs):
            check(); q = pidx[j]; x, y = q//width, q % width
            old = tokens.flatten(1).gather(1, q).clone()
            oracle_old = oracle.flatten(1).gather(1, q).clone()
            neighbors = torch.stack([tokens[rows, x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]], -1)
            oracle_neighbors = torch.stack([oracle[rows, x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]], -1)
            target = torch.sigmoid(2*float(c.BETA)*(2*neighbors.float()-1).sum(-1))
            oracle_target = torch.sigmoid(2*float(c.BETA)*(2*oracle_neighbors.float()-1).sum(-1))
            masked = tokens.clone(); masked.flatten(1).scatter_(1, q, 2)
            mt = torch.full((n,), max(q.shape[1]/width**2, clock), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(masked, mt, xy, valid)
            prob = logits.float().softmax(1)[:, 1].flatten(1).gather(1, q)
            new = (pu[j] < prob).long()
            oracle_new = (pu[j] < oracle_target).long()
            tokens.flatten(1).scatter_(1, q, new)
            oracle.flatten(1).scatter_(1, q, oracle_new)
            prob_trace.append(prob); target_trace.append(target); oracle_target_trace.append(oracle_target)
            old_trace.append(old); new_trace.append(new); oracle_old_trace.append(oracle_old); oracle_new_trace.append(oracle_new)
            neighbor_trace.append(neighbors); oracle_neighbor_trace.append(oracle_neighbors)
            calls += 1
        result.update(prefix_spins=(2*prefix.cpu().numpy()-1).astype(np.int8),
                      oracle_spins=(2*oracle.cpu().numpy()-1).astype(np.int8), repair_positions=positions.transpose(1, 0, 2),
                      repair_uniforms=uniforms.transpose(1, 0, 2), repair_rng_seeds=repair_seeds,
                      repair_model_t=np.array(max(positions.shape[-1]/width**2, clock)))
        for name, series in [("repair_probability", prob_trace), ("repair_target", target_trace),
                             ("oracle_target", oracle_target_trace), ("repair_old", old_trace), ("repair_new", new_trace),
                             ("oracle_old", oracle_old_trace), ("oracle_new", oracle_new_trace),
                             ("repair_neighbors", neighbor_trace), ("oracle_neighbors", oracle_neighbor_trace)]:
            result[name] = torch.stack(series, 1).cpu().numpy()
    result["spins"] = (2*tokens.cpu().numpy()-1).astype(np.int8)
    result["network_calls"] = np.array(calls)
    assert calls == total_steps and np.isin(result["spins"], [-1, 1]).all()
    return result


def audit_repair(result):
    if "prefix_spins" not in result:
        return
    n, w, _ = result["spins"].shape
    rows = np.arange(n)[:, None]
    for prefix, final, old_key, new_key, neighbors_key, probability_key in [
            ("prefix_spins", "spins", "repair_old", "repair_new", "repair_neighbors", "repair_probability"),
            ("prefix_spins", "oracle_spins", "oracle_old", "oracle_new", "oracle_neighbors", "oracle_target")]:
        state = ((result[prefix]+1)//2).copy()
        for j in range(result["repair_positions"].shape[1]):
            q = result["repair_positions"][:, j]
            x, y = q//w, q % w
            assert np.all((x > 0) & (x < w-1) & (y > 0) & (y < w-1))
            assert np.all((x+y) % 2 == j % 2)
            assert all(len(np.unique(z)) == len(z) for z in q)
            assert np.array_equal(state.reshape(n, -1)[rows, q], result[old_key][:, j])
            actual_neighbors = np.stack([state[rows, x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]], -1)
            assert np.array_equal(actual_neighbors, result[neighbors_key][:, j])
            p = 1/(1+np.exp(-2*c.BETA*(2*actual_neighbors.astype(float)-1).sum(-1)))
            target_key = "repair_target" if final == "spins" else "oracle_target"
            assert np.allclose(p, result[target_key][:, j], atol=1e-7, rtol=0)
            assert np.array_equal((result["repair_uniforms"][:, j] < result[probability_key][:, j]).astype(int), result[new_key][:, j])
            state.reshape(n, -1)[rows, q] = result[new_key][:, j]
        assert np.array_equal(2*state-1, result[final])
