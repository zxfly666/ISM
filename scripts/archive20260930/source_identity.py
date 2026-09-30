"""Compare recorded frozen Python source hashes with the public source checkout."""
import hashlib
import json
from pathlib import Path
from catalog import ROOT,C

def hashb(b):return hashlib.sha256(b).hexdigest()
def main():
    rows=[]
    for c in C:
        base=ROOT/'experiments'/c['slug']/'evidence'
        for p in base.glob('*protocol*.json'):
            x=json.loads(p.read_text(encoding='utf-8-sig'))
            for key in ('sources','files','source_sha256','source_hashes'):
                mapping=x.get(key)
                if not isinstance(mapping,dict):continue
                for name,v in mapping.items():
                    if not name.endswith('.py'):continue
                    expected=v.get('sha256') if isinstance(v,dict) else v
                    if not isinstance(expected,str) or len(expected)!=64:continue
                    local=ROOT/name;status='not_present';actual=None
                    if not local.is_file() and '/' not in name:
                        candidates=list((ROOT/'ism_diffusion').glob(name))+list((ROOT/'tests').glob(name))+list((ROOT/'scripts').glob('*/'+name))
                        matching=[q for q in candidates if hashb(q.read_bytes())==expected]
                        if len(matching)==1:local=matching[0]
                    if local.is_file():
                        b=local.read_bytes();actual=hashb(b)
                        status='exact' if actual==expected else 'line_endings_only' if hashb(b.replace(b'\r\n',b'\n'))==expected else 'different_version'
                    rows.append(dict(campaign=c['slug'],protocol=p.name,path=name,resolved_path=local.relative_to(ROOT).as_posix(),expected_sha256=expected,current_sha256=actual,status=status))
    out=ROOT/'experiments/source-identity-audit.json'
    out.write_text(json.dumps(dict(scope='Frozen Python entries discoverable in published root protocol hash maps; not a claim that every historical source map was present.',records=rows),indent=2)+'\n',encoding='utf-8')
    from collections import Counter
    print(json.dumps(dict(count=len(rows),status=dict(Counter(r['status'] for r in rows)),mismatches=[r for r in rows if r['status']=='different_version']),ensure_ascii=False))

if __name__=='__main__':main()
