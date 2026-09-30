"""Unchanged dense architecture, deterministic updates and complete recovery."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

import sm6_common as c
import sm6_data as d

sys.path.insert(0, str(c.ROOT))
sys.path.insert(0, str(c.ROOT / 'scripts/research20260927'))
from intervention_model import InterventionDenoiser
from ism_diffusion.scale_model import CoordinateDenoiserConfig


def configure():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def new_model(seed, device):
    torch.manual_seed(c.init_seed(seed))
    cfg = c.CFG['model']
    kwargs = {k: cfg[k] for k in ('d_model', 'n_heads', 'n_blocks', 'mlp_ratio', 'dropout', 'vocab_size', 'output_classes', 'rope_base')}
    model = InterventionDenoiser(CoordinateDenoiserConfig(**kwargs), 'dense').to(device)
    assert sum(p.numel() for p in model.parameters()) == cfg['parameters']
    return model


def fresh(seed, device):
    model = new_model(seed, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    cfg = c.CFG['training']['optimizer']
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=tuple(cfg['betas']),
                           weight_decay=cfg['weight_decay'], foreach=False)
    return model, ema, opt


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_bank(bank, device):
    types = dict(tokens=torch.long, coordinates=torch.float32, valid=torch.bool,
                 t=torch.float32, target=torch.float32, query=torch.long)
    return {k: torch.as_tensor(bank[k], dtype=kind, device=device) for k, kind in types.items()}


def lr(step, total=8000, warmup=512):
    assert 1 <= step <= total
    if step <= warmup:
        return 3e-4*step/warmup
    return 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def update_parts(state, parts, step, total=8000, warmup=512, keep_gradient=False):
    model, ema, opt = state
    model.train()
    device = next(model.parameters()).device
    count = sum(len(p['query']) for p in parts)
    opt.zero_grad(set_to_none=True)
    losses = []
    for p in parts:
        tb = tensor_bank(p, device)
        logits = model(tb['tokens'], tb['t'], tb['coordinates'], tb['valid'])
        query_logits = logits.flatten(2).transpose(1, 2)[torch.arange(len(p['query']), device=device), tb['query']]
        lp = query_logits.log_softmax(-1)
        y = tb['target']
        loss = -(y*lp[:, 1]+(1-y)*lp[:, 0])
        assert bool(torch.isfinite(loss).all())
        (loss.sum()/count).backward()
        losses.append(loss.detach())
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert bool(torch.isfinite(grad))
    gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None} if keep_gradient else None
    rate = lr(step, total, warmup)
    for group in opt.param_groups:
        group['lr'] = rate
    opt.step()
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(.999).add_(p, alpha=.001)
    combined = torch.cat(losses).cpu().numpy()
    result = dict(step=int(step), loss=float(combined.mean()), grad_norm=float(grad), lr=rate,
                  family_losses=[float(v.mean()) for v in np.split(combined, 4)] if count == 128 else [float(combined.mean())])
    return (result, gradients) if keep_gradient else result


def update(state, data, seed, step, arm, log=None):
    started = time.perf_counter()
    c.deadline()
    parts, identity = data.batch(seed, step, arm)
    result = update_parts(state, parts, step)
    result.update(identity)
    result.update(seed=seed, arm=arm, time=time.time(), seconds=time.perf_counter()-started)
    if log is not None:
        import json
        log.write(json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n')
    return result


@torch.inference_mode()
def predict(model, bank, batch=32):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(bank['query']), batch):
        c.deadline(required=False)
        b = d.subset(bank, np.arange(start, min(start+batch, len(bank['query']))))
        tb = tensor_bank(b, device)
        logits = model(tb['tokens'], tb['t'], tb['coordinates'], tb['valid'])
        q = logits.flatten(2).transpose(1, 2)[torch.arange(len(b['query']), device=device), tb['query']]
        result.append(q.cpu().numpy())
    values = np.concatenate(result).astype(np.float64)
    assert np.isfinite(values).all()
    return values


def checkpoint(path, state, seed, arm, step, protocol_hash, full=True, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    model, ema, opt = state
    payload = dict(study=c.CFG['study'], config_sha=c.sha(c.CONFIG_PATH), seed=seed, arm=arm,
                   step=step, next_step=step+1, protocol_hash=protocol_hash,
                   raw=model.state_dict(), ema=ema.state_dict(), time=time.time(),
                   raw_hash=model_hash(model), ema_hash=model_hash(ema),
                   model_parameters=sum(p.numel() for p in model.parameters()))
    payload.update(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   torch_cpu_rng=torch.get_rng_state(),
                   torch_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   data_counter=step, data_rng='PCG64/SeedSequence addressed by root, seed, step, role; no arm')
    if full:
        payload.update(optimizer=opt.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_cpu_rng=torch.get_rng_state(),
                       torch_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       data_counter=step, data_rng='PCG64/SeedSequence addressed by root, seed, step, role; no arm')
    temp = path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:
        torch.save(payload, f)
    os.replace(temp, path)
    return {k: payload[k] for k in ('seed', 'arm', 'step', 'raw_hash', 'ema_hash', 'model_parameters')}


def restore(path, device):
    # AdamW non-capturable step counters are CPU tensors. Loading every storage
    # directly to CUDA wrongly migrates those counters; load_state_dict then
    # preserves their wrong placement. Stage on CPU and let the optimizer move
    # parameter-shaped moments according to its policy.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['study'] == c.CFG['study'] and payload['config_sha'] == c.sha(c.CONFIG_PATH)
    state = fresh(payload['seed'], device)
    state[0].load_state_dict(payload['raw'])
    state[1].load_state_dict(payload['ema'])
    state[2].load_state_dict(payload['optimizer'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_cpu_rng'].cpu())
    if str(device).startswith('cuda') and payload['torch_cuda_rng']:
        torch.cuda.set_rng_state_all([x.cpu() for x in payload['torch_cuda_rng']])
    assert model_hash(state[0]) == payload['raw_hash'] and model_hash(state[1]) == payload['ema_hash']
    assert state[2].state and all(int(v['step']) == payload['step'] for v in state[2].state.values())
    assert all(bool(torch.isfinite(v).all()) for s in state[2].state.values() for v in s.values() if torch.is_tensor(v))
    return state, payload


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b
