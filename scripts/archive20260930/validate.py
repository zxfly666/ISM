"""Validate public evidence bytes, array schemas, links and publication hygiene.

No training, no network, no checkpoint deserialization. Optional --receipt writes
only an archive QA record. Error messages do not print possible secret values.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def public_paths():
    a=read(OUT/'publication-allowlist.json')
    return sorted(set(list(OUT.rglob('*'))+[ROOT/p for p in a['scientific_sources']+a['effective_config_sources']]+list((ROOT/'scripts/archive20260930').glob('*.py'))+[ROOT/'README.md',ROOT/'ARCHIVE_PLAN.md',ROOT/'.gitattributes']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt');args=ap.parse_args()
    errors=[];count=0;npzs=0;array_count=0;nonfinite=[];bytes_total=0;max_file=(0,'');link_count=0
    for p in OUT.rglob('provenance.json'):
        for item in read(p)['files']:
            target=OUT/item['published']
            if not target.is_file() or target.stat().st_size!=item['bytes'] or digest(target)!=item['sha256']:
                errors.append({'kind':'provenance mismatch','path':str(target.relative_to(ROOT))})
            count+=1
    for item in read(OUT/'source-manifest.json'):
        if digest(ROOT/item['path'])!=item['sha256']:errors.append({'kind':'source changed','path':item['path']})
    secret=[re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
            re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{35,}|AKIA[A-Z0-9]{16})\b'),
            re.compile(r'(?i)["\']?(?:password|passwd|api_key|access_token|client_secret)["\']?\s*[:=]\s*["\'][A-Za-z0-9/+_-]{16,}["\']')]
    for p in public_paths():
        if not p.is_file():continue
        size=p.stat().st_size;bytes_total+=size
        if size>max_file[0]:max_file=(size,p.relative_to(ROOT).as_posix())
        if size>50*1024**2:errors.append({'kind':'oversized ordinary-git file','path':p.relative_to(ROOT).as_posix(),'bytes':size})
        if p.suffix=='.npz':
            npzs+=1
            try:
                with np.load(p,allow_pickle=False) as z:
                    for k in z.files:
                        a=z[k];array_count+=1
                        if a.dtype.kind in 'fc' and not np.all(np.isfinite(a)):
                            nonfinite.append({'path':p.relative_to(ROOT).as_posix(),'array':k,'count':int(np.sum(~np.isfinite(a)))})
            except Exception as e:errors.append({'kind':'NPZ unreadable without pickle','path':p.relative_to(ROOT).as_posix(),'error':type(e).__name__})
        if p.suffix not in ('.py','.md','.json','.csv','.txt','.jsonl'):continue
        try:t=p.read_text(encoding='utf-8-sig')
        except UnicodeError:errors.append({'kind':'text encoding','path':p.relative_to(ROOT).as_posix()});continue
        for pat in secret:
            if pat.search(t):errors.append({'kind':'possible credential; review without echo','path':p.relative_to(ROOT).as_posix()})
        if p.suffix=='.json':
            try:json.loads(t)
            except ValueError:errors.append({'kind':'JSON invalid','path':p.relative_to(ROOT).as_posix()})
        if p.suffix=='.py':
            try:ast.parse(t)
            except SyntaxError:errors.append({'kind':'Python syntax','path':p.relative_to(ROOT).as_posix()})
        if p.suffix=='.md':
            t=re.sub(r'```.*?```','',t,flags=re.S)
            for m in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',t):
                target=m.group(1).strip('<>');target=target.split(' "')[0]
                if target.startswith(('http:','https:','mailto:','#')):continue
                link_count+=1; path=unquote(target.split('#')[0])
                if path and not (p.parent/path).exists():errors.append({'kind':'broken local link','path':p.relative_to(ROOT).as_posix(),'target':target})
    result=dict(passed=not errors,provenance_records=count,npz_files=npzs,arrays=array_count,
                public_bytes=bytes_total,largest_file={'bytes':max_file[0],'path':max_file[1]},
                local_links_checked=link_count,nonfinite_arrays_preserved=nonfinite,errors=errors,
                limitations=['Nonfinite diagnostic values are inventoried, not silently removed or treated as universally invalid.','Secret scan is pattern-based plus manual payload review, not a proof of absence.','Links checked against local public paths; staged-only closure is checked separately.'])
    if args.receipt:
        p=ROOT/args.receipt;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='nonfinite_arrays_preserved'},ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
