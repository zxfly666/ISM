"""Fixed J18h GPU checks and timings; no formal train/evaluation starts here."""
from pathlib import Path
import copy
import gc
import json
import math
import os
import platform
import resource
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
from ism_diffusion.scale_data import load_parent_split, ParentSplit
import intervention_common as c
import intervention_data as d
import intervention_training as tr
from intervention_scratch import sliced


def configure():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set deterministic CUBLAS environment before process start')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def exact(a, b):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            raise RuntimeError('Non-exact restored tensor')
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise RuntimeError('Non-exact restored array')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise RuntimeError('Restored dictionary keys differ')
        for k in a: exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b): raise RuntimeError('Restored sequence lengths differ')
        for x, y in zip(a, b): exact(x, y)
    elif a != b:
        raise RuntimeError('Restored scalar differs')


def equivalence_and_recovery(out, parent, check):
    fixture = d.fixture_bank()
    x = torch.as_tensor(fixture['noisy'][:2], device='cuda', dtype=torch.long)
    xy = torch.as_tensor(fixture['input_coordinates'][:2], device='cuda')
    t = torch.as_tensor(fixture['t'][:2], device='cuda')
    diagnostics = []
    for index in range(6):
        check()
        ident = c.identity('continuation', index, 'D')
        m, e, o, meta = tr.initialize(ident, 'cuda', c.base_checkpoint(index))
        old = CoordinateDenseDenoiser(CoordinateDenoiserConfig(**c.MODEL)).cuda().eval()
        old.load_state_dict(m.state_dict())
        if list(dict(old.named_parameters())) != list(dict(m.named_parameters())):
            raise RuntimeError('Parameter registration order changed')
        with torch.inference_mode():
            a, b = old(x, t, xy), m.eval()(x, t, xy)
            exact(a, b)
        maxima = {}
        for mode in ('dense', 'observed_only'):
            m.attention_mode = mode
            with torch.inference_mode():
                base = m(x, t, xy).softmax(1)[:, 1].flatten(1)
                for shift in ((7, -11), (-13, 5)):
                    z = m(x, t, xy+torch.tensor(shift, device='cuda')).softmax(1)[:, 1].flatten(1)
                    maxima[mode+str(shift)] = float((z-base).abs().max())
                perm = torch.randperm(64, device='cuda'); inv = torch.argsort(perm)
                z = m(x.flatten(1)[:, perm, None].transpose(1, 2), t,
                      xy.reshape(2, 64, 2)[:, perm, None].transpose(1, 2)).softmax(1)[:, 1].flatten(1)[:, inv]
                maxima[mode+'_permutation'] = float((z-base).abs().max())
                z = torch.cat([m(x[j:j+1], t[j:j+1], xy[j:j+1]).softmax(1)[:, 1].flatten(1) for j in range(2)])
                maxima[mode+'_batch'] = float((z-base).abs().max())
                pad_x = F.pad(x.reshape(2, 1, 64), (0, 16), value=3)
                pad_xy = F.pad(xy.reshape(2, 1, 64, 2), (0, 0, 0, 16), value=0)
                valid = pad_x != 3
                z = m(pad_x, t, pad_xy, valid).softmax(1)[:, 1].flatten(1)[:, :64]
                maxima[mode+'_PAD'] = float((z-base).abs().max())
                empty = x.clone(); empty[0] = 2
                z = m(empty, t, xy).softmax(1)
                oldz = old(empty[:1], t[:1], xy[:1]).softmax(1)
                maxima[mode+'_K0'] = float((z[:1]-oldz).abs().max())
        if max(maxima.values()) > 2e-5:
            raise RuntimeError(('GPU FP32 equivalence failed', maxima))
        diagnostics.append(dict(base=index, dense_exact=True, checks=maxima,
            raw=meta['initial_raw_hash'], ema=meta['initial_ema_hash'], optimizer_restored_step=12000))
        del m, e, o, old
    c.write(out/'gpu_equivalence.json', dict(status='passed', bases=diagnostics, tolerance=2e-5))
    # Fixed small native-width batch, full token budget. The deterministic
    # numerical contract is exact, not a retrospectively chosen tolerance.
    torch.use_deterministic_algorithms(True)
    records = []
    for mode in ('D', 'O'):
        ident = c.identity('continuation', 0, mode)
        j = next(s for s in range(1, 33) if d.schedule(ident['data_seed'], s, 4000)['width'] == 16)
        batch = d.training_batch(parent, ident, j)
        m, e, o, meta = tr.initialize(ident, 'cuda', c.base_checkpoint(0))
        with sdpa_kernel(SDPBackend.MATH):
            row = tr.update(m, e, o, batch, 'continuation', 1, amp=False)
            meta.update(step=1, global_step=12001)
            file = out/f'recovery_{mode}.pt'
            tr.checkpoint(file, m, e, o, meta, 'software_recovery')
            expected = tr.update(m, e, o, batch, 'continuation', 2, amp=False)
            m2, e2, o2, _ = tr.restore(file, ident, 'software_recovery', 'cuda')
            observed = tr.update(m2, e2, o2, batch, 'continuation', 2, amp=False)
            exact(expected, observed); exact(m.state_dict(), m2.state_dict())
            exact(e.state_dict(), e2.state_dict()); exact(o.state_dict(), o2.state_dict())
            if mode == 'D':
                old = CoordinateDenseDenoiser(CoordinateDenoiserConfig(**c.MODEL)).cuda()
                p = torch.load(file, map_location='cpu', weights_only=False)
                old.load_state_dict(p['model']); eo = copy.deepcopy(old).eval().requires_grad_(False)
                eo.load_state_dict(p['ema']); oo = tr.optimizer(old); oo.load_state_dict(p['optimizer'])
                rr = tr.update(old, eo, oo, batch, 'continuation', 2, amp=False)
                exact(rr, observed); exact(old.state_dict(), m2.state_dict())
                exact(eo.state_dict(), e2.state_dict()); exact(oo.state_dict(), o2.state_dict())
                del old, eo, oo, p
        records.append(dict(mode=mode, restored_model_ema_optimizer_exact=True, old_dense_exact=mode=='D', row=expected))
        del m, e, o, m2, e2, o2; torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)
    c.write(out/'recovery.json', dict(status='passed', contract='exact FP32 math SDPA', records=records))


def capability(out, check):
    b = d.fixture_bank(); models = {}; results = []
    c.save(out/'fixture_input.npz', **b)
    x = torch.as_tensor(b['noisy'], device='cuda', dtype=torch.long)
    xy = torch.as_tensor(b['input_coordinates'], device='cuda')
    t = torch.as_tensor(b['t'], device='cuda')
    y = torch.as_tensor(b['labels'][:, 0], device='cuda', dtype=torch.float32)
    initial = None
    for mode in ('dense', 'observed_only'):
        m = tr.new_model(c.CONFIG['seeds']['fixture'], mode, 'cuda')
        h = gs.model_hash(m)
        if initial is not None and initial != h: raise RuntimeError('Fixture initialization mismatch')
        initial = h; o = tr.optimizer(m); trace = []
        for step in range(1, 1025):
            check(); o.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = m(x, t, xy)[:, :, 3, 3]
            lp = logits.float().log_softmax(1)
            loss = -(y*lp[:, 1]+(1-y)*lp[:, 0]).mean()
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite fixture loss')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            if not bool(torch.isfinite(grad)): raise RuntimeError('Nonfinite fixture gradient')
            lr = 3e-4*step/64 if step <= 64 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-64)/960))
            for group in o.param_groups: group['lr'] = lr
            o.step(); trace.append((step, float(loss.detach()), float(grad), lr))
        pred = tr.predict(m, b, check=check)
        maximum = float(pred['kl'].max())
        c.save(out/f'fixture_{mode}_predictions.npz', **pred)
        c.save(out/f'fixture_{mode}_trace.npz', trace=np.array(trace))
        with (out/f'fixture_{mode}.pt').open('xb') as f:
            torch.save(m.state_dict(), f)
        result = dict(mode=mode, seed=c.CONFIG['seeds']['fixture'], steps=1024,
            initial_hash=h, max_pattern_kl=maximum, passed=maximum <= .01,
            final_raw_not_ema=True, no_adaptive_retries=True)
        c.write(out/f'fixture_{mode}.json', result); results.append(result)
        models[mode] = m.eval(); del o
    passed = all(x['passed'] for x in results)
    c.write(out/'capability.json', dict(status='passed' if passed else 'failed', results=results))
    if not passed:
        raise RuntimeError('Fixed 1024-step capability gate failed; no retry/extension')
    return models


def measure(call, check):
    call()  # exactly one untimed warmup, still charged to elapsed E
    rows = []
    for _ in range(3):
        check(); torch.cuda.synchronize(); began = time.perf_counter()
        call(); torch.cuda.synchronize(); rows.append(time.perf_counter()-began)
    return dict(samples_seconds=rows, maximum_seconds=max(rows), warmups=1, fixed_repetitions=3)


def timings(out, parent, models, check):
    rows = []; T = 0.
    # One fixed first cycle gives all 32 cells, including all ordinary repeats.
    for mode in ('D', 'O'):
        ident = c.identity('fresh', 0, mode)
        m = tr.new_model(771029, ident['attention_mode'], 'cuda')
        e = copy.deepcopy(m).eval().requires_grad_(False); o = tr.optimizer(m)
        times = []
        for step in range(1, 33):
            def run_step():
                b = d.training_batch(parent, ident, step)
                tr.update(m, e, o, b, 'fresh', step)
            z = measure(run_step, check); z.update(mode=mode, step=step, **d.schedule(ident['data_seed'], step, 24000))
            rows.append(z); times.append(z['maximum_seconds'])
        T += sum(times)/32*168000
        del m, e, o; torch.cuda.empty_cache()
    c.write(out/'training_timing.json', dict(rows=rows, projected_training_seconds=T))
    # Native inference of each N/K/clock family; use OLD training fields only.
    import intervention_pipeline as pipe
    from intervention_scratch import run as scratch
    small = ParentSplit(parent.spins[:4], np.array([0, 0, 1, 1]), 1024, {})
    folders = out/'timing_banks'; folders.mkdir(); pipe.copy_blueprint(folders)
    pipe.make_banks(small, folders, check, scratch=True)  # bank-building warmup
    bank_rows=[]
    for rep in range(3):
        path=out/f'timing_banks_{rep}'; path.mkdir();pipe.copy_blueprint(path)
        pipe.make_banks(small,path,check,scratch=True)
        bank_rows.append(c.read(path/'banks/complete.json')['build_timings'])
    scales={x['name']:x['parents']/4 for x in c.CONFIG['core_banks']}
    scales.update(mechanism=64,padding=64,oracle=16,low=1)
    bank_projection=sum(max(row[k] for row in bank_rows)*scale for k,scale in scales.items())
    c.write(out/'bank_timing.json',dict(samples=bank_rows,scales=scales,projected_seconds=bank_projection))
    validation = d.validation_banks(load_parent_split(c.DATA, 'val'))
    infer = []; Fsec = 0.; Vsec = 0.; io = []; byte_prediction = 0
    sizes = dict(core=None, mechanism=256, padding=256, oracle=1024, low=960, validation=64)
    entries = [('core', p.stem, c.load(p)) for p in (folders/'banks').glob('*.npz') if not p.stem.endswith('_cpu_baselines')]
    entries += [(cat, p.stem, c.load(p)) for cat in ('mechanism', 'padding', 'oracle', 'low') for p in (folders/'diagnostic_banks'/cat).glob('*.npz')]
    entries += [('validation', name, b) for name, b in validation.items()]
    for category, name, b in entries:
        n = next(x['parents'] for x in c.CONFIG['core_banks'] if x['name']==name) if category=='core' else sizes[category]
        # Each timing forward contains two actual native-sized inputs.
        sample = sliced(b, 2)
        for mode, model in models.items():
            result = measure(lambda: tr.predict(model, sample, check=check), check)
            result.update(category=category, name=name, mode=mode, inputs=2, native_shape=list(sample['noisy'].shape))
            infer.append(result)
            # Dense: six B0 + twelve D finals; O: twelve O finals.
            mult = (18 if mode=='dense' else 12)+(6 if category=='core' else 0)
            if category=='validation':
                Vsec += result['maximum_seconds']/2*64*48
            else:
                Fsec += result['maximum_seconds']/2*n*mult
        # Measure every actual native result-file shape, including metadata.
        raw = tr.predict(models['dense'], sample, check=check)
        synthetic = {key: np.resize(value, (n,)+value.shape[1:]) if isinstance(value, np.ndarray) and value.ndim and len(value)==2 else value
                     for key, value in raw.items()}
        # Do not benchmark unrealistically compressible tiled probabilities.
        random = c.rng(71425, n, category+'_'+name+'_IO_ONLY')
        synthetic['probability'] = random.uniform(.001, .999, (n,)+raw['probability'].shape[1:])
        if 'labels' in synthetic:
            y = synthetic['labels']; p = synthetic['probability']
            synthetic['ce'] = -(y*np.log(p)+(1-y)*np.log1p(-p))
            synthetic['brier'] = y*(1-p)**2+(1-y)*p**2
            if 'kl' in synthetic:
                synthetic['kl'] = synthetic['ce']+y*np.log(y)+(1-y)*np.log1p(-y)
                synthetic['probability_error'] = p-y
        synthetic['timing_only_not_scientific'] = np.array(True)
        counter = [0]
        def output():
            p = out/'io'/f'{category}_{name}_{counter[0]}.npz'; counter[0] += 1
            c.save(p, **synthetic)
            with p.open('rb') as f: os.fsync(f.fileno())
        z = measure(output, check)
        mult = 96 if category=='validation' else (42 if category=='core' else 30)
        io.append(dict(category=category, name=name, multiplicity=mult, **z))
        byte_prediction += (out/'io'/f'{category}_{name}_1.npz').stat().st_size*mult
    # Save/load full optimizer checkpoint at an actual trained state.
    ident = c.identity('continuation', 0, 'D')
    m,e,o,meta = tr.initialize(ident,'cuda',c.base_checkpoint(0)); meta.update(step=1,global_step=12001)
    b = d.training_batch(parent, ident, 1); tr.update(m,e,o,b,'continuation',1)
    j = [0]
    def checkpoint_io():
        p=out/f'timing_checkpoint_{j[0]}.pt'; j[0]+=1
        tr.checkpoint(p,m,e,o,meta,'timing')
        restored = tr.restore(p,ident,'timing','cuda')
        del restored
    cp = measure(checkpoint_io,check); T += Vsec + cp['maximum_seconds']*(336+24+48)
    del m,e,o; torch.cuda.empty_cache()
    c.write(out/'inference_io_timing.json',dict(inference=infer,writing=io,checkpoint=cp,
        training_with_validation_and_checkpoint_seconds=T, formal_inference_seconds=Fsec,
        prediction_write_seconds=sum(x['maximum_seconds']*x['multiplicity'] for x in io if x['category']!='validation'),
        validation_write_seconds=sum(x['maximum_seconds']*x['multiplicity'] for x in io if x['category']=='validation'),
        projected_prediction_bytes=byte_prediction))
    T += sum(x['maximum_seconds']*x['multiplicity'] for x in io if x['category']=='validation')
    # Full actual-predictor small-sample plumbing and ten test plots.
    began=time.perf_counter(); scratch(out/'software_scratch_gpu',check,models=models)
    c.write(out/'scratch_seconds.json',dict(seconds=time.perf_counter()-began))
    return dict(T=T,F=Fsec+bank_projection,
        bank_build_projected_seconds=bank_projection,
        L=sum(x['maximum_seconds']*x['multiplicity'] for x in io if x['category']!='validation'))


def cpu_timings(out, parent, check):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from intervention_mc_timing import worker
    import intervention_statistics as st
    mc=[]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        for repeat in range(3):
            check()
            results=list(pool.map(worker,[(71829+repeat*4+j,parent.spins[k]) for j,k in enumerate((0,255,511,767))]))
            mc.append(results)
    # Four workers, four chains each; 10 burn-equivalent intervals and 256 production intervals.
    mc_seconds=max(9782.3692,max(z['seconds'] for row in mc for z in row)*4*(256+10+1))
    layouts=c.load(c.ROOT/'artifacts/geometry_identification_20260926/design/low_layouts.npz')
    subset=ParentSplit(parent.spins[:32],np.repeat(np.arange(16),2),1024,{})
    began=time.perf_counter(); low=d.low_reference(subset,layouts,check)
    low_seconds=(time.perf_counter()-began)*128
    # Benchmark original-parent QA energy/m/G crop computation, not new confirmation sampling.
    from ism_diffusion.ising import energy_density,magnetization
    from evaluate_study import physical_axis_statistics
    qa=[]
    for _ in range(3):
        began=time.perf_counter()
        energy_density(subset.spins);magnetization(subset.spins)
        axis=np.broadcast_to(np.arange(96),(32,2,96)).copy()
        crop=2*d.sampled(subset,np.arange(32),axis,np.zeros((32,2),dtype=int))-1
        physical_axis_statistics(crop,[axis[:,0],axis[:,1]])
        qa.append(time.perf_counter()-began)
    mc_seconds += low_seconds+max(qa)*128
    chain=np.repeat(np.arange(16),256)
    random=np.random.default_rng(817720)
    risk=random.normal(.5,.05,(6,7,2,4096))
    began=time.perf_counter();st.resample([risk[:,:3],risk[:,3:]],chain,256,8,'timing',check=check)
    core_seconds=(time.perf_counter()-began)/256*(2*60000+2*60000/16)
    grouping,_=st.geometry_groups(layouts)
    models=np.full((6,5,80,2,6),.5)
    counts=np.tile(low['counts'],(128,1,1,1))
    began=time.perf_counter();st.low_bootstrap(counts,chain,grouping,models,20,check)
    low_stats=(time.perf_counter()-began)/20*2000
    c.write(out/'cpu_timing.json',dict(mc_timing=mc,qa_timing=qa,low_table_seconds=low_seconds,
        projected_M=mc_seconds,projected_core_statistics_seconds=core_seconds,projected_low_statistics_seconds=low_stats,
        old_fields_only=True,formal_MC_unchanged=True))
    return dict(M=mc_seconds,A=core_seconds+low_stats+600)


def run(out, check):
    out=Path(out); out.mkdir(exist_ok=False)
    configure(); check()
    c.write(out/'environment.json',dict(time=time.time(),python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        threads=torch.get_num_threads(),cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],TF32=False,
        env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        torch_config=torch.__config__.show()))
    parent=load_parent_split(c.DATA,'train')
    print('EQUIVALENCE_AND_RECOVERY',flush=True); equivalence_and_recovery(out,parent,check)
    print('FIXED_CAPABILITY',flush=True); models=capability(out,check)
    print('NATIVE_TIMINGS_AND_GPU_SCRATCH',flush=True); a=timings(out,parent,models,check)
    del models; torch.cuda.empty_cache();gc.collect()
    print('MC_QA_STATISTICS_TIMINGS',flush=True); a.update(cpu_timings(out,parent,check))
    a['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    a['process_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c.write(out/'computed_checks.json',dict(status='passed_computation_needs_visual_transfer_budget_gate',**a))
    return a
