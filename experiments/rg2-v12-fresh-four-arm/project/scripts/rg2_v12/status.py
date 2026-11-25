"""Small read-only status snapshot for the active A/B campaign."""
import json
import os
from pathlib import Path
import shutil
import time

root = Path(__file__).resolve().parents[2]
runs = root / 'runs'
if (runs/'campaign_B/status.json').exists():
    state = json.loads((runs/'campaign_B/status.json').read_text())
    current = json.loads((runs/'B_v2/status.json').read_text()) if (runs/'B_v2/status.json').exists() else {}
    count = len(list((runs/'B_v2').glob('s*/complete.json')))
    samples = len(list((runs/'B_eval/samples').rglob('*.npz')))
    print(f"B {state['status']} {state.get('stage','')} models={count}/8 images={samples}/512")
    if current: print(f"seed={current['seed']} arm={current['arm']} step={current['step']}/12000")
else:
    state = json.loads((runs/'campaign/status.json').read_text())
    counts = [len(list((runs/p).rglob('*.npz'))) for p in ('A_supp/A0_retention','A_supp/A2','A_v2/samples')]
    print(f"A {state['status']} {state['stage']} retention={counts[0]}/735 A2={counts[1]}/396 images={counts[2]}/960")
pid = state.get('worker_pid',state.get('pid'))
if pid:
    try: os.kill(pid,0); alive=True
    except ProcessLookupError: alive=False
    print(f"worker_alive={alive}")
print(f"free_GiB={shutil.disk_usage(root).free/1024**3:.2f} utc={time.strftime('%H:%M:%S',time.gmtime())}")
