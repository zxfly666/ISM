"""Create public copies from an explicit local allowlist, without executing science.

Run only with custodian sources present. Never download or modify an original.
All byte changes in copied text have a provenance ledger. Numerical NPZs are
byte-for-byte copies, except explicitly named lossless bootstrap projections.
"""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
import numpy as np
from catalog import ROOT, DOC, BACKUP, C, SOURCE_DIRS, PRIVATE_CODE

DEST = ROOT / 'experiments'
TEXT = {'.json','.csv','.md','.txt','.jsonl'}
LIMIT = 8 * 1024**2
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def local_label(p):
    p = Path(p)
    try: return p.relative_to(ROOT).as_posix()
    except ValueError:
        try: return 'custodian-backups/'+p.relative_to(BACKUP).as_posix()
        except ValueError: return p.name

def public_text(b, markdown=False):
    t=b.decode('utf-8-sig'); changes=[]
    substitutions=[
        (r'(?i)[A-Z]:[/\\]Users[/\\][^/\\\s"<>]+','<LOCAL_USER>', 'local username'),
        (r'219\.146\.211\.42|ssh\.bj8\.bz1\.paratera\.com|p-6fbac01d114f-ackcs-00gjhlaj|ackcs-00gjhlaj','<REDACTED_HOST>','connection identity'),
        (r'(?i)D:/ISM_research_backups','<CUSTODIAN_BACKUPS>','backup root'),
        (r'(?i)D:/ISM|/root/shared-nvme/ISM_research_20260921','<PROJECT_ROOT>','machine-specific project root'),
    ]
    for pattern,repl,why in substitutions:
        t,n=re.subn(pattern,repl,t)
        if n: changes.append({'kind':why,'count':n})
    if markdown:
        # Historical relative references are source identifiers, not false public links.
        def link(m):
            label,target=m.groups()
            if target.startswith(('https://','http://','#')): return m.group()
            return label+' (`historical source: '+target+'`)'
        t,n=re.subn(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',link,t)
        if n: changes.append({'kind':'historical local link rendered as source identifier','count':n})
        t='> Public archival copy. Historical planning/status text is preserved; consult the campaign README for actual execution and final decisions. Infrastructure identifiers are redacted. Relative links below are historical source identifiers. Original SHA and every transformation are recorded in provenance.json.\n\n'+t
        changes.append({'kind':'public-copy notice'})
    # JSON structure must remain parseable after infrastructure redaction.
    return t.encode('utf-8'),changes

def save_copy(p,dest,records):
    raw=p.read_bytes(); out=raw; transforms=[]
    if p.suffix in TEXT:
        out,transforms=public_text(raw,p.suffix=='.md')
        if p.suffix=='.json': json.loads(out)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(out)
    records.append(dict(source=local_label(p),published=dest.relative_to(DEST).as_posix(),
                        original_bytes=len(raw),original_sha256=sha(raw),
                        bytes=len(out),sha256=sha(out),transformations=transforms))

ROOT_SKIP={'status.json','processes.json','launch_status.json','budget.json','authorization.json',
           'last_training_status.json','input_freeze.json','legacy_freeze_check.json','source_context_v1.verification.json'}
def allowed(p):
    if p.name in ('launch_gate.json','run_environment.json','environment.json','runtime.json'):
        return p.stat().st_size<=LIMIT
    return (p.suffix in {'.json','.csv','.npz','.png','.pdf'} and p.stat().st_size<=LIMIT
            and not any(s in p.name for s in ('monitor_','scratch','smoke','launch','dispatch','queue','process','environment'))
            and p.name not in ROOT_SKIP)

def rootsmall(src,target,records):
    if not src.exists(): raise FileNotFoundError(src)
    for p in sorted(src.iterdir()):
        if p.is_file() and allowed(p): save_copy(p,target/p.name,records)

def tree(src,target,records):
    if not src.exists(): return
    for p in sorted(src.rglob('*')):
        if p.is_file() and allowed(p): save_copy(p,target/p.relative_to(src),records)

def normalize_manifest(x):
    if isinstance(x,list):
        return {v['path']:v for v in x if isinstance(v,dict) and 'path' in v and 'sha256' in v}
    if isinstance(x,dict) and isinstance(x.get('files'),list):
        return {v['path']:v for v in x['files'] if isinstance(v,dict) and 'path' in v and 'sha256' in v}
    if isinstance(x,dict) and isinstance(x.get('files'),dict): x=x['files']
    if isinstance(x,dict): return {k:v for k,v in x.items() if isinstance(v,dict) and 'sha256' in v}
    return {}

def run():
    allrec=[]; registry=[]
    for c in C:
        base=DEST/c['slug']; src=Path(c['source'])
        if not src.is_absolute():src=ROOT/src
        rec=[]; rootsmall(src,base/'evidence',rec)
        if c['analysis']:tree(src/c['analysis'],base/'evidence'/c['analysis'],rec)
        for nm in c['protocols']:save_copy(DOC/nm,base/'protocol'/nm,rec)
        for source,target in c['extra']:
            p=ROOT/source
            if p.is_file():save_copy(p,base/target,rec)
            elif p.is_dir():
                rootsmall(p,base/target,rec)
                tree(p/'figures',base/target/'figures',rec)
            else:raise FileNotFoundError(p)
        tree(src/'figures',base/'figures',rec)
        tree(src/'reference',base/'evidence/reference',rec)
        # Read-only exact local outputs: all final predictions, not weights or preflight fixtures.
        if c['slug'] in ('exact-local-capability','local-context-size-diagnostic'):
            tree(src/'evaluation',base/'evidence/evaluation',rec)
            for p in sorted((src/'training').rglob('*')):
                if p.is_file() and allowed(p):save_copy(p,base/'evidence/training'/p.relative_to(src/'training'),rec)
        # Keep all official plots together in their original relative layout.
        # Large bootstrap containers are projected losslessly, not recomputed.
        if c['slug']=='size-spacing-factorial':
            for p in sorted((src/'analysis').glob('conditional_*.npz')):
                if p.stat().st_size<=LIMIT:continue
                with np.load(p,allow_pickle=False) as z:
                    keys=[k for k in z.files if k!='arm_means']
                    values={k:z[k] for k in keys}
                target=base/'evidence/analysis'/(p.stem+'_primary_projection.npz')
                np.savez_compressed(target,**values)
                rec.append(dict(source=local_label(p),published=target.relative_to(DEST).as_posix(),
                                original_bytes=p.stat().st_size,original_sha256=sha(p.read_bytes()),
                                bytes=target.stat().st_size,sha256=sha(target.read_bytes()),
                                transformations=[{'kind':'lossless NPZ named-array extraction','retained_keys':keys,'omitted_keys':['arm_means']}]))
        manifests={}; archives=[]; receipts=[]
        for nm in ['manifest.json','science_manifest.json','closing_science_manifest_v1.json','final_analysis/manifest.json']:
            p=src/nm
            if p.exists():manifests.update(normalize_manifest(read(p)))
        if c['backup']:
            bp=BACKUP/c['backup']
            for p in sorted(bp.glob('*.manifest.json')):
                x=read(p); entries=normalize_manifest(x)
                archives.append(dict(manifest=p.name,archive=x.get('archive',p.name.replace('.manifest.json','.tar.gz')),
                                     archive_bytes=x.get('archive_bytes'),archive_sha256=x.get('archive_sha256'),members=len(entries)))
                for path,v in entries.items():
                    if c['experiment_id'] in path or 'parents_l1024' in path:
                        entry=dict(v);entry['archive_manifest']=p.name
                        manifests.setdefault(path,entry)
            for p in sorted(bp.glob('*verification*.json')):
                # Archive receipts assert local verification, not a public download.
                receipts.append({'receipt':p.name,'sha256':sha(p.read_bytes()),'result':read(p)})
        for p in sorted(src.rglob('*')):
            if p.is_file() and p.suffix in ('.pt','.pth','.npz','.jsonl') and p.stat().st_size>LIMIT:
                rel=p.relative_to(src).as_posix()
                if rel not in manifests: manifests[rel]={'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
        public_hash={r['original_sha256']:r['published'] for r in rec}
        for path,v in manifests.items():
            v['public_copy']=public_hash.get(v.get('sha256'))
            v['availability']='public copy (see provenance for redaction)' if v['public_copy'] else 'custodian-held; not a public download'
        write(base/'withheld-manifest.json',dict(note='Contains scientific identity records including public copies. Local verification does not imply public availability. Request archive/member by SHA through a repository issue; no hosting or retention guarantee is made.',archives=archives,files=manifests))
        rb,_=public_text(json.dumps(receipts,ensure_ascii=False).encode())
        write(base/'backup-verification.json',json.loads(rb))
        write(base/'provenance.json',dict(policy='Original evidence unchanged. Public text redactions are explicit; no numerical result changes.',files=rec))
        row={k:c[k] for k in ['slug','public_name','historical_alias','experiment_id','date','training_kind','parents','new_reference','new_generation','code']}
        row['formal_completed']=not c['slug'].startswith('preflights/')
        row['published_files']=len(rec);row['published_bytes']=sum(r['bytes'] for r in rec)
        registry.append(row);allrec+=rec
        print(json.dumps({'campaign':c['slug'],'files':len(rec),'MB':round(row['published_bytes']/1e6,2),'manifest_entries':len(manifests)}))
    write(DEST/'registry.json',registry)
    # Include source code by explicit scientific-directory allowlist, not git add -A.
    sources=[]
    for folder in SOURCE_DIRS:
        for p in sorted((ROOT/folder).glob('*.py')):
            if p.name not in PRIVATE_CODE:sources.append(p)
    sources += [ROOT/'ism_diffusion/geometry_study.py',ROOT/'ism_diffusion/context_repair_core.py']
    sources += list((ROOT/'tests').glob('test_*geometry*.py'))+list((ROOT/'tests').glob('test_diagnostic_math.py'))+list((ROOT/'tests').glob('test_mechanism_design.py'))
    source_rec=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sources]
    write(DEST/'source-manifest.json',source_rec)
    # Configs retain original repo paths because frozen Python imports use them.
    configs=sorted(set(nm for c in C for nm in c['protocols'] if nm.endswith('.json')))
    tracked_inputs=['docs/research_reboot_20260921/'+nm for nm in configs]
    write(DEST/'publication-allowlist.json',dict(scientific_sources=[v['path'] for v in source_rec],effective_config_sources=tracked_inputs,
                                               archive_root='experiments',excluded=['credentials','connection guides','private correspondence','checkpoints','bulk MC/generation','unrelated working-tree files']))
    print('Total copied scientific evidence bytes:',sum(r['bytes'] for r in allrec))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true',required=True);p.parse_args();run()
