"""Read-only schema and source inspection; never runs a scientific pipeline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def inspect(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as z:
            return {k: {'shape': z[k].shape, 'dtype': str(z[k].dtype)} for k in z.files}
    if p.suffix == '.json':
        x = json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(x, dict):
            return {'keys_count': len(x), 'head': {k: (list(v)[:12] if isinstance(v, dict)
                    else 'list[%d]' % len(v) if isinstance(v, list) else v)
                    for k, v in list(x.items())[:45]}}
        return {'list': len(x), 'first': x[:1]}
    if p.suffix == '.py':
        x = ast.parse(p.read_text(encoding='utf-8-sig'))
        return {'imports': [ast.dump(n) for n in ast.walk(x) if isinstance(n, (ast.Import, ast.ImportFrom))],
                'functions': [n.name for n in ast.walk(x) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]}
    raise ValueError(p)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('paths', nargs='+'); args = ap.parse_args()
    for path in args.paths:
        print(json.dumps({'path': path, 'schema': inspect(path)}, ensure_ascii=False))
