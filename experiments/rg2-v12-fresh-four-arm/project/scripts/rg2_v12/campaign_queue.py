"""Detached, restartable stage-A queue. Later stages require a separate gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import phase_a as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs'))
    args=p.parse_args();root=args.root.absolute();out=root/'campaign';out.mkdir(parents=True,exist_ok=True)
    jobs=[
        ('retention',['stage_a_supplement.py','retention','--out',str(root/'A_supp')]),
        ('exact_blocks',['stage_a_supplement.py','a2','--out',str(root/'A_supp')]),
        ('generation',['phase_a.py','a1','--device','cuda','--n','32','--out',str(root/'A_v2')]),
    ]
    frozen=dict(protocol='RG-2 v1.2',code={str(Path(__file__).parent/name):a.file_sha(Path(__file__).parent/name)
                for name in ('phase_a.py','stage_a_supplement.py','campaign_queue.py')},jobs=jobs,
                inputs=a.file_sha(a.ROOT/'inputs/manifest.json'),supplement_inputs=a.file_sha(root/'A_supp/inputs.json'),
                stage_B_authorized_by_this_queue=False)
    state=dict(pid=os.getpid(),started=time.time(),stage='waiting_existing_A0_A2',status='running')
    with a.run_lock(out/'campaign.lock'):
        path=out/'frozen.json'
        if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(frozen))
        else:a.write_json(path,frozen)
        a.write_json(out/'status.json',state)
        while not (root/'A_v2/diagnostics.exit').exists():time.sleep(5)
        if (root/'A_v2/diagnostics.exit').read_text().strip()!='0':raise RuntimeError('A0/A2 preliminary diagnostics failed')
        if (root/'A_supp/prepare.exit').read_text().strip()!='0':raise RuntimeError('Supplement preparation failed')
        for name,argv in jobs:
            marker=out/(name+'.json')
            if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':continue
            if shutil.disk_usage(a.ROOT).free<1024**3:raise RuntimeError('Less than 1 GiB disk headroom')
            for path,expected in frozen['code'].items():assert a.file_sha(Path(path))==expected
            state.update(stage=name,status='running',stage_started=time.time());a.write_json(out/'status.json',state)
            cmd=[sys.executable,'-u',str(Path(__file__).parent/argv[0]),*argv[1:]]
            print('START',name,flush=True)
            with (out/(name+'.log')).open('ab') as log:
                proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                state['worker_pid']=proc.pid;a.write_json(out/'status.json',state);code=proc.wait()
            result=dict(status='complete' if code==0 else 'failed',exit_code=code,seconds=time.time()-state['stage_started'],finished=time.time())
            a.write_json(marker,result)
            if code:
                state.update(status='failed',exit_code=code);a.write_json(out/'status.json',state);return code
            print('DONE',name,result['seconds'],flush=True)
        state.update(status='stage_A_computation_complete_review_required',finished=time.time())
        a.write_json(out/'status.json',state);print(state['status'],flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
