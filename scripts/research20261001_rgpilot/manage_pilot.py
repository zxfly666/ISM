"""CPU-only observation and final administrative-byte closure. Never launches science."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import common as c

def processes():
    rows=[]
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args=(path/'cmdline').read_bytes().decode(errors='replace').replace('\x00',' ')
            if 'research20261001_rgpilot/run_pilot.py' not in args:continue
            # Ignore shell observers mentioning the string as data.
            name=(path/'comm').read_text().strip()
            if name not in ['python','python3','timeout']:continue
            pid=int(path.name)
            detail=subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,pgid,etime,%cpu,%mem,rss,stat,comm','--no-headers'],capture_output=True,text=True).stdout.strip()
            rows.append(dict(pid=pid,process=detail))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return rows

def monitor():
    proto=c.read(c.OUT/'run_protocol.json');c.check_sources(proto)
    result=dict(time=time.time(),processes=processes(),sources='passed',status=None,failure=None,
        compute_deadline=c.CFG['compute_deadline'],delivery_deadline=c.CFG['delivery_deadline'],
        disk_free_bytes=shutil.disk_usage(c.ROOT).free,
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip())
    for key,path in [('status','status.json'),('failure','failure.json'),('complete','final_summary.json'),('export','remote_export_complete.json')]:
        if (c.OUT/path).exists():result[key]=c.read(c.OUT/path)
    total=0;tails={};finite=True
    for path in c.OUT.glob('training/*/log.jsonl'):
        lines=path.read_text().splitlines();rows=[]
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            rows.append(row)
            finite=finite and __import__('math').isfinite(row['loss']) and __import__('math').isfinite(row['grad'])
        total+=len(rows)
        if rows:tails[path.parent.name]=rows[-1]
    result.update(updates=total,total_updates=c.CFG['total_updates'],all_recorded_training_finite=finite,
        tails=tails,finals=len(list(c.OUT.glob('training/*/complete.json'))),
        prediction_files=len(list((c.OUT/'predictions').rglob('*.npz'))))
    for p in c.OUT.glob('training/*/complete.json'):
        assert c.sha(p.parent/'final.pt')==c.read(p)['final_sha256']
    dest=c.EXPORT/('monitor_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'.json')
    c.write(dest,result)
    print(json.dumps(dict(path=str(dest),**result),ensure_ascii=False))

def closure():
    assert not processes(),'Scientific process still running'
    assert (c.OUT/'remote_export_complete.json').exists()
    proto=c.read(c.OUT/'run_protocol.json');c.check_sources(proto)
    science=c.read(c.OUT/'science_manifest.json')
    for row in science['files']:
        path=c.ROOT/row['path'];assert path.stat().st_size==row['bytes'] and c.sha(path)==row['sha256']
    late=[c.OUT/n for n in ['formal.stdout','status.json','run.jsonl','remote_export_complete.json','science_manifest.json'] if (c.OUT/n).exists()]
    combined={r['path']:r for r in science['files']}
    for row in c.records(late):combined[row['path']]=row
    dest=c.EXPORT/'closure_manifest_v1.json'
    c.write(dest,dict(files=list(combined.values()),time=time.time(),status='bytes_closed_needs_local_union',no_scientific_recomputation=True))
    a=c.archive(c.ROOT,c.EXPORT,'rgpilot_administrative_closure_v1',late+[dest])
    print(json.dumps(dict(archive=a['archive'],bytes=a['archive_bytes'],members=a['members'])))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['monitor','closure'])
    {'monitor':monitor,'closure':closure}[ap.parse_args().mode]()
