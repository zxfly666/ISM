"""Detached A→B gated development queue. This program never starts C."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import phase_a as a

def main():
    root=a.ROOT/'runs';out=root/'campaign_B';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((root/'A_to_B_gate.json').read_text())
    assert gate['A_to_B']=='accepted' and gate['C_to_be_decided_after_B'] is True
    assert json.loads((root/'B_v2/preflight.json').read_text())['status']=='passed'
    steps=[('training',['phase_b.py','train','--out','runs/B_v2','--gate','runs/A_to_B_gate.json']),
           ('evaluation',['phase_b_evaluate.py','--training','runs/B_v2','--out','runs/B_eval'])]
    frozen=dict(gate_sha256=a.file_sha(root/'A_to_B_gate.json'),
                code={str(Path(__file__).parent/n):a.file_sha(Path(__file__).parent/n) for n in
                      ('phase_a.py','phase_b.py','phase_b_evaluate.py','campaign_b.py')},steps=steps)
    with a.run_lock(out/'queue.lock'):
        if (out/'frozen.json').exists():assert json.loads((out/'frozen.json').read_text())==json.loads(json.dumps(frozen))
        a.write_json(out/'frozen.json',frozen)
        for name,argv in steps:
            marker=out/(name+'.json')
            if marker.exists() and json.loads(marker.read_text())['status']=='complete':continue
            if shutil.disk_usage(a.ROOT).free<2*1024**3:raise RuntimeError('Insufficient space for new checkpoints and local archive')
            for path,h in frozen['code'].items():assert a.file_sha(Path(path))==h
            state=dict(stage=name,status='running',pid=os.getpid(),started=time.time(),C_launched=False)
            with (out/(name+'.log')).open('ab') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/argv[0]),*argv[1:]],
                                      cwd=a.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                state['worker_pid']=proc.pid;a.write_json(out/'status.json',state);code=proc.wait()
            result=dict(status='complete' if code==0 else 'failed',exit_code=code,seconds=time.time()-state['started'])
            a.write_json(marker,result)
            if code:
                state.update(status='failed',exit_code=code);a.write_json(out/'status.json',state);return code
        a.write_json(out/'status.json',dict(status='B_complete_review_required',finished=time.time(),C_launched=False))
    return 0

if __name__=='__main__':sys.exit(main())
