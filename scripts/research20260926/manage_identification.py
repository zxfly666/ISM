"""Read-only science monitor and completed-branch, content-hashed backups.

Only writes sibling exports or the explicitly selected local verification receipt.
Never starts science, changes its outputs, deletes files or overwrites a package.
"""
import argparse,datetime,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
import identification_common as c

EXPORT=c.ROOT/'artifacts/geometry_identification_exports_20260926'

def latest(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-10000));lines=f.read().decode('utf-8',errors='replace').splitlines()
    for s in reversed(lines):
        try:return json.loads(s)
        except (ValueError,TypeError):pass
    return None

def monitor():
    result=dict(time=time.time(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training={},failures={})
    for key in ['budget','status','processes','final_summary']:
        p=c.OUT/(key+'.json');result[key]=c.read(p) if p.exists() else None
    result['process_table']=subprocess.run(['ps','-eo','pid,ppid,pgid,stat,pcpu,pmem,etime,args'],capture_output=True,text=True).stdout
    result['process_table']='\n'.join(x for x in result['process_table'].splitlines() if 'research20260926' in x or 'multiprocessing' in x)
    result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout
    result['disk']=subprocess.run(['df','-B1',str(c.ROOT)],capture_output=True,text=True).stdout
    for p in (c.OUT/'training').glob('s*'):
        result['training'][p.name]=dict(last=latest(p/'train.jsonl'),complete=(p/'complete.json').exists())
    result['counts']=dict(training_complete=len(list((c.OUT/'training').glob('*/complete.json'))),evaluation_complete=len(list((c.OUT/'evaluation').glob('*/complete.json'))),predictions=len(list((c.OUT/'evaluation').glob('*/conditional/*.npz'))),shards=len(list((c.OUT/'evaluation').glob('*/generation/shard_*.npz'))),banks=len(list((c.OUT/'banks').glob('*.npz'))))
    for name in ['failure.json','reference_failure.json','reference/qa.json','reference/complete.json']:
        p=c.OUT/name
        if p.exists():result['failures' if 'failure' in name else 'training'][name]=c.read(p)
    if (c.OUT/'run_protocol.json').exists():
        protocol=c.read(c.OUT/'run_protocol.json');result['source_mismatch']=[p for p,h in protocol['files'].items() if c.sha(c.ROOT/p)!=h]
        result['run_protocol_sha256']=c.sha(c.OUT/'run_protocol.json')
        result['final_mismatch']=[]
        for p in (c.OUT/'training').glob('*/complete.json'):
            if c.sha(p.parent/'final.pt')!=c.read(p)['final_sha256']:result['final_mismatch'].append(p.parent.name)
    if result['budget']:result['remaining_hours']=(result['budget']['deadline']-time.time())/3600
    EXPORT.mkdir(exist_ok=True);path=EXPORT/('monitor_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    c.write(path,result,exclusive=True);print(json.dumps(dict(path=str(path),counts=result['counts'],remaining=result.get('remaining_hours'),status=result['status'],failures=result['failures'],source_mismatch=result.get('source_mismatch'),gpu=result['gpu'])))

def export(stage,cell,name):
    if not name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in name):raise ValueError('Safe unique name required')
    if stage in ['training','evaluation']:
        if cell not in {f's{s}_{a}' for s in c.SEEDS for a in c.ARMS}:raise ValueError('Unknown cell')
        folder=c.OUT/stage/cell;assert c.read(folder/'complete.json')['status']=='complete'
        if stage=='training':assert c.sha(folder/'final.pt')==c.read(folder/'complete.json')['final_sha256']
        paths=list(folder.rglob('*'))
    elif stage in ['reference','banks']:
        folder=c.OUT/stage;assert (folder/'complete.json').exists();paths=list(folder.rglob('*'))
    elif stage=='initial':
        assert (c.OUT/'run_protocol.json').exists();paths=list(c.source_files())
        for folder in ['preflight_v1','cpu_tests','design']:paths+=list((c.OUT/folder).rglob('*'))
        paths+=list(c.OUT.glob('*.json'))
    elif stage=='final':
        assert (c.OUT/'final_summary.json').exists();paths=list((c.OUT/'analysis').rglob('*'))+list((c.OUT/'padding_banks').rglob('*'))+list((c.OUT/'validation_banks').rglob('*'))+list(c.OUT.glob('*'))
    else:raise ValueError(stage)
    paths=sorted(set(p for p in paths if p.is_file() and p.name not in ['last.pt','restore_fixture.pt','timing.pt'] and not p.name.endswith('.tmp')))
    EXPORT.mkdir(exist_ok=True);archive=EXPORT/(name+'.tar.gz');manifest=EXPORT/(name+'.manifest.json')
    if archive.exists() or manifest.exists():raise FileExistsError(name)
    rows=[]
    with archive.open('xb') as file:
        with tarfile.open(fileobj=file,mode='w:gz',compresslevel=2) as tar:
            for p in paths:
                rel=p.relative_to(c.ROOT).as_posix();h=c.sha(p);rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=h));tar.add(p,arcname=rel,recursive=False)
                assert c.sha(p)==h
    data=dict(stage=stage,cell=cell,files=rows,archive=archive.name,bytes=archive.stat().st_size,sha256=c.sha(archive),members=len(rows))
    c.write(manifest,data,exclusive=True);print(json.dumps({k:v for k,v in data.items() if k!='files'}))

def verify(archive,manifest,receipt):
    ar=Path(archive);m=c.read(manifest);assert c.sha(ar)==m['sha256'] and ar.stat().st_size==m['bytes']
    rows={x['path']:x for x in m['files']};seen=set()
    with tarfile.open(ar,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in rows and member.name not in seen
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            seen.add(member.name);h=hashlib.sha256();f=tar.extractfile(member)
            for buf in iter(lambda:f.read(4*1024**2),b''):h.update(buf)
            assert member.size==rows[member.name]['bytes'] and h.hexdigest()==rows[member.name]['sha256']
    assert seen==set(rows)
    c.write(receipt,dict(status='passed',archive=str(ar.resolve()),archive_sha256=m['sha256'],verified_members=len(seen),time=time.time()),exclusive=True)
    print('VERIFIED',len(seen),m['sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['monitor','export','verify']);p.add_argument('--stage');p.add_argument('--cell');p.add_argument('--name');p.add_argument('--archive');p.add_argument('--manifest');p.add_argument('--receipt');a=p.parse_args()
    if a.mode=='monitor':monitor()
    elif a.mode=='export':export(a.stage,a.cell,a.name)
    else:verify(a.archive,a.manifest,a.receipt)
