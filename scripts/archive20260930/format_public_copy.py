"""Remove redundant EOF blank lines only in derived public Markdown protocols.

Never changes original protocols/frozen code; update the explicit editorial ledger.
"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'experiments'
def main():
    changes=[]
    for p in E.rglob('provenance.json'):
        x=json.loads(p.read_text(encoding='utf-8'));changed=False
        for r in x['files']:
            f=E/r['published']
            if f.suffix!='.md':continue
            b=f.read_bytes()
            if not b.endswith((b'\n\n',b'\r\n\r\n')):continue
            assert hashlib.sha256(b).hexdigest()==r['sha256']
            new=b.rstrip(b'\r\n')+b'\n';f.write_bytes(new)
            r['bytes']=len(new);r['sha256']=hashlib.sha256(new).hexdigest()
            r['transformations'].append({'kind':'remove redundant EOF blank lines from public Markdown copy only'})
            changed=True;changes.append(r['published'])
        if changed:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'public_copies_formatted':changes}))
if __name__=='__main__':main()
