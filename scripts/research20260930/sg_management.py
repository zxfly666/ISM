"""Non-GPU auditing, safe member archives, union SHA coverage and monitoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path
import numpy as np
import sg_common as c
import sg_data as d


def audit_training():
    data=d.TrainingData(); records=[]; first={}; finals={}
    for seed in c.CFG['seed_labels']:
        streams={a:(c.OUT/'training'/f's{seed}_{a}'/'log.jsonl').open(encoding='utf-8') for a in 'ABCD'}
        try:
            for step in range(1,24001):
                if step%1000==0: c.deadline()
                current=[]
                for arm,handle in streams.items():
                    row=json.loads(next(handle)); assert row['seed']==seed and row['arm']==arm and row['step']==step
                    assert np.isfinite([row['loss'],row['grad_norm'],*row['family_losses']]).all()
                    _,expected=data.batch(seed,step,arm)
                    for key in expected: assert row[key]==expected[key],(seed,arm,step,key)
                    current.append(row)
                assert len({v['paired_data_digest'] for v in current})==1
            assert all(handle.read()=='' for handle in streams.values())
        finally:
            for handle in streams.values():handle.close()
        for arm in 'ABCD':
            folder=c.OUT/'training'/f's{seed}_{arm}'
            init=c.read(folder/'initial.json'); first[seed,arm]=init['raw_hash']
            complete=c.read(folder/'complete.json'); assert complete['step']==24000
            assert c.sha(folder/'final.pt')==complete['final_sha256']
            finals[f's{seed}_{arm}']=complete['final_sha256']
        assert len({first[seed,a] for a in 'ABCD'})==1
    return dict(status='passed',updates_reconstructed=576000,physical_and_native_input_hashes_recomputed=True,
                all_finite=True,initialization_pairs=6,final_sha=finals,time=time.time())


def audit_evaluation(root, rows, fixture=False):
    count=0;paired={};source_cache={}
    for row in rows:
        z=c.load(root/row['prediction']); b=c.load(root/row['bank'])
        assert np.array_equal(z['target'],b['target']) and np.array_equal(z['row_id'],b['row_id'])
        d.validate_view(b)
        assert str(z['native_input_hash'])==d.native_digest([b])==row['native_hash']
        common=c.digest(b['row_id'],b['tokens'],b['coordinates'],b['valid'],b['query'],b['target'])
        assert str(z['common_input_hash'])==common
        group=(row['seed'],row['step'],row['weight'],row['spec']['key'])
        if group in paired:assert paired[group]==common
        paired[group]=common
        assert str(z['source_sha256'])==row['source_sha256']
        if fixture:
            assert row['source_sha256']=='SYNTHETIC_FIXTURE_NOT_A_CHECKPOINT'
        else:
            path=root/'training'/f"s{row['seed']}_{row['arm']}"/('final.pt' if row['step']==24000 else f"step_{row['step']}.pt")
            if path not in source_cache:source_cache[path]=c.sha(path)
            assert source_cache[path]==row['source_sha256']
        expected=c.metrics(z['logits'],z['target'])
        for key,v in expected.items(): assert np.allclose(v,z[key],rtol=0,atol=2e-12)
        summary=c.summary(z['logits'],z['target'])
        for key,value in summary.items(): assert abs(float(value)-float(row[key]))<2e-12
        count+=1
    return dict(status='passed',predictions_recomputed=count,common_input_pairing_groups=len(paired),
                checkpoint_sources_checked=len(source_cache),fixture=fixture,time=time.time())


def archive(root, export, name, paths):
    export.mkdir(parents=True,exist_ok=True)
    tarpath=export/(name+'.tar.gz'); manifestpath=export/(name+'.manifest.json')
    if tarpath.exists() or manifestpath.exists(): raise FileExistsError(name)
    paths=sorted(set(Path(p) for p in paths))
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)) for p in paths]
    with tarfile.open(tarpath,'x:gz',compresslevel=3,dereference=True) as tar:
        for p in paths:tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    manifest=dict(archive=tarpath.name,archive_sha256=c.sha(tarpath),archive_bytes=tarpath.stat().st_size,
                  members=len(files),files=files,created=time.time())
    c.write(manifestpath,manifest)
    return manifest


def verify(archive_path,manifest_path,receipt):
    path=Path(archive_path); manifest=c.read(manifest_path)
    assert c.sha(path)==manifest['archive_sha256']
    expected={r['path']:r for r in manifest['files']}; seen=set()
    with tarfile.open(path,'r:gz') as tar:
        for entry in tar:
            assert entry.isfile() and entry.name in expected and entry.name not in seen
            assert not entry.name.startswith('/') and '..' not in Path(entry.name).parts
            h=hashlib.sha256()
            with tar.extractfile(entry) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
            assert h.hexdigest()==expected[entry.name]['sha256'] and entry.size==expected[entry.name]['bytes']
            seen.add(entry.name)
    assert seen==set(expected)
    result=dict(status='passed',time=time.time(),archive=path.name,archive_bytes=path.stat().st_size,
                archive_sha256=manifest['archive_sha256'],manifest_sha256=c.sha(manifest_path),members=len(seen))
    c.write(receipt,result)
    return result


def union(manifest_path,backup_dir,receipt):
    science=c.read(manifest_path); covered={}; packages=[]
    for rec in sorted(Path(backup_dir).glob('*.verification.json')):
        r=c.read(rec)
        if r.get('status')!='passed':continue
        name=r['archive'].removesuffix('.tar.gz')
        mpath=Path(backup_dir)/(name+'.manifest.json'); apath=Path(backup_dir)/r['archive']
        assert c.sha(mpath)==r['manifest_sha256'] and c.sha(apath)==r['archive_sha256']
        mm=c.read(mpath); packages.append(apath.name)
        for row in mm['files']: covered[(row['path'],row['sha256'],row['bytes'])]=apath.name
    missing=[row for row in science['files'] if (row['path'],row['sha256'],row['bytes']) not in covered]
    result=dict(status='passed' if not missing else 'failed',time=time.time(),science_paths=len(science['files']),
                covered=len(science['files'])-len(missing),missing=missing,packages=packages,
                manifest_sha256=c.sha(manifest_path),same_disk_not_offsite=True)
    c.write(receipt,result)
    assert not missing
    return result


def package_science():
    proto=c.read(c.OUT/'run_protocol.json'); c.check_sources(proto['sources'])
    paths=[p for p in c.OUT.rglob('*') if p.is_file() and p.name!='last.pt' and not any(x in p.parts for x in ('__pycache__','preflight','cpu_preparation'))]
    paths += [c.ROOT/p for p in proto['sources']]
    # The formal scientific manifest is made before packaging; it is not self-referential.
    science=dict(files=[dict(path=p.relative_to(c.ROOT).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(set(paths))],
                 exclusions=[p.relative_to(c.ROOT).as_posix() for p in c.OUT.rglob('last.pt')],time=time.time())
    c.write(c.OUT/'science_manifest.json',science)
    paths += [c.OUT/'science_manifest.json']
    return archive(c.ROOT,c.EXPORT,'complete_science_v1',paths)


def monitor():
    import subprocess
    result=dict(time=time.time(),resources=c.resources(),status=c.read(c.OUT/'status.json') if (c.OUT/'status.json').exists() else None,
                budget=c.read(c.OUT/'budget.json') if (c.OUT/'budget.json').exists() else None,
                failures={p.name:c.read(p) for p in c.OUT.glob('*failure*.json')},
                completed_training=len(list(c.OUT.glob('training/*/complete.json'))),
                final_summary=(c.OUT/'final_summary.json').exists())
    if os.name!='nt':
        p=subprocess.run(['ps','-eo','pid,ppid,pgid,etime,%cpu,args'],capture_output=True,text=True)
        result['processes']=[line for line in p.stdout.splitlines() if 'research20260930' in line]
    result['tails']={}
    for path in c.OUT.glob('training/*/log.jsonl'):
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-4096)); lines=f.read().splitlines()
        if lines:
            try:result['tails'][path.parent.name]=json.loads(lines[-1])
            except json.JSONDecodeError:pass
    if (c.OUT/'run_protocol.json').exists():
        c.check_sources(c.read(c.OUT/'run_protocol.json')['sources']);result['source_hashes']='passed'
    dest=c.EXPORT/('monitor_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'.json')
    c.write(dest,result)
    return dict(path=str(dest),**result)


if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('monitor')
    v=sub.add_parser('verify');v.add_argument('--archive',required=True);v.add_argument('--manifest',required=True);v.add_argument('--receipt',required=True)
    u=sub.add_parser('union');u.add_argument('--manifest',required=True);u.add_argument('--backup-dir',required=True);u.add_argument('--receipt',required=True)
    a=p.parse_args()
    value=monitor() if a.mode=='monitor' else verify(a.archive,a.manifest,a.receipt) if a.mode=='verify' else union(a.manifest,a.backup_dir,a.receipt)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))
