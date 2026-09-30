"""Report CPU/static preparation separately from outstanding GPU gates."""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=['D:/ISM/scripts/research20260921']
import core_geometry_factorial_statistics as statistics

root=Path('D:/ISM/scripts/research20260921')
names=['core_geometry_factorial_training.py','core_geometry_factorial_statistics.py']
for name in names:ast.parse((root/name).read_text(encoding='utf-8'))
result=dict(status='cpu_statistics_and_static_training_checked',
    conditional_bootstrap=statistics.self_test(),
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
    trained=False,gpu_tested=False,resume_tested=False,budget_clock_started=False,
    remaining=['runner','MC_preparation','generation','full_statistics_and_figures',
               'GPU_update_and_resume_tests','full_scratch_pipeline','runtime_gate'])
target=Path(__file__).with_name('cpu_preparation_20260925_0348.json')
target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
