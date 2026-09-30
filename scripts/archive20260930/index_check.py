"""Read-only audit of exactly the Git index/publication delta, not untracked files."""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import unquote
from validate import public_paths, ROOT, OUT

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt');args=p.parse_args()
    entries={}
    for line in git('ls-files','--stage','-z').split(b'\0'):
        if not line:continue
        meta,path=line.split(b'\t',1);mode,oid,stage=meta.split()
        assert stage==b'0','unmerged index'
        entries[path.decode('utf-8')]=oid.decode()
    errors=[];expected=[f for f in public_paths() if f.is_file()]
    for f in expected:
        rel=f.relative_to(ROOT).as_posix()
        if rel not in entries:errors.append({'kind':'public file absent from index','path':rel});continue
        # Byte identity is required for scientific evidence and scientific source hashes.
        if rel.startswith('experiments/') or rel in [v['path'] for v in json.loads((OUT/'source-manifest.json').read_text())]:
            b=f.read_bytes();oid=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
            if oid!=entries[rel]:errors.append({'kind':'index bytes differ from public file','path':rel})
        if f.suffix!='.md':continue
        t=re.sub(r'```.*?```','',f.read_text(encoding='utf-8-sig'),flags=re.S)
        for m in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',t):
            target=m.group(1).strip('<>').split(' "')[0]
            if target.startswith(('http:','https:','mailto:','#')):continue
            path=unquote(target.split('#')[0]);dest=(f.parent/path).resolve()
            try:r=dest.relative_to(ROOT).as_posix()
            except ValueError:errors.append({'kind':'link outside public repo','path':rel,'target':target});continue
            if r not in entries and not any(q.startswith(r.rstrip('/')+'/') for q in entries):
                errors.append({'kind':'link target not in index','path':rel,'target':target})
    delta=git('diff','--cached','--name-only','--diff-filter=ACMR','ed2235990fd213fcfaad30e6bde7480b08780c30','-z').split(b'\0')
    delta=[v.decode() for v in delta if v]; total=0;maxsize=0;credential_hits=[];archives=[]
    pat=re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{25,}\b|github_pat_[A-Za-z0-9_]{35,}')
    for name in delta:
        # Local bytes are checked against index for the main payload above; this also
        # audits the two earlier unpushed commits' small source/preflight packages.
        f=ROOT/name;b=f.read_bytes() if f.is_file() else b''
        local_oid=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if local_oid!=entries[name]:b=git('cat-file','blob',entries[name])
        total+=len(b);maxsize=max(maxsize,len(b))
        if len(b)>50*1024**2:errors.append({'kind':'delta file exceeds 50MiB','path':name})
        if name.endswith(('.pt','.pth')):errors.append({'kind':'unexpected checkpoint payload','path':name})
        if pat.search(b):credential_hits.append(name)
        if name.endswith('.tar.gz'):
            with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
                members=[]
                for ent in tf:
                    if not ent.isfile():continue
                    raw=tf.extractfile(ent).read()
                    if pat.search(raw):credential_hits.append(name+'#'+ent.name)
                    members.append(ent.name)
                archives.append({'path':name,'members':members})
    errors += [{'kind':'possible credential (value suppressed)','path':x} for x in credential_hits]
    result=dict(passed=not errors,expected_public_files=len(expected),indexed_files=len(entries),
                publication_delta_files=len(delta),publication_delta_bytes=total,largest_delta_bytes=maxsize,
                small_archives_inspected=archives,errors=errors,
                scope='Exact index compared to last verified remote main. No staging/committing/pushing performed by this check.')
    if args.receipt:(ROOT/args.receipt).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='small_archives_inspected'},ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
