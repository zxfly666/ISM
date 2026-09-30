"""Publish all latest exact-task predictions/controls and their input banks.

Selective member reads from an already verified local archive; no tar extraction,
no model execution. Enforces member size and the original SHA manifest.
"""
import json
import tarfile
from pathlib import Path
from catalog import ROOT, BACKUP
from curate import sha, public_text, write

def main():
    bp=BACKUP/'20260930_dense_multisize_canonical_6h'
    m=json.loads((bp/'complete_science_v1.manifest.json').read_text())
    archive=bp/'complete_science_v1.tar.gz'
    assert sha(archive.read_bytes())==m['archive_sha256']
    members={v['path']:v for v in m['files']}
    prefix='artifacts/dense_multisize_canonical_6h_20260930/'
    base=ROOT/'experiments/canonical-context-size-generalization'
    prov=json.loads((base/'provenance.json').read_text()); records={v['published']:v for v in prov['files']}
    count=0
    with tarfile.open(archive,'r|gz') as tf:
        for ent in tf:
            if not ent.isfile() or not ent.name.startswith(prefix):continue
            rel=ent.name[len(prefix):]
            if not (rel.startswith(('banks/','evaluation/','controls/')) or rel=='evaluation_manifest.json'):continue
            if Path(rel).suffix not in ('.npz','.json') or ent.size>8*1024**2:continue
            assert '..' not in Path(rel).parts
            raw=tf.extractfile(ent).read(); ident=members[ent.name]
            assert len(raw)==ident['bytes'] and sha(raw)==ident['sha256'],ent.name
            out=raw; transforms=[]
            if rel.endswith('.json'):out,transforms=public_text(raw)
            dest=base/'evidence'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(out)
            key=dest.relative_to(ROOT/'experiments').as_posix()
            records[key]=dict(source='custodian-backups/'+bp.name+'/'+archive.name+'#'+ent.name,
                source_archive_sha256=m['archive_sha256'],published=key,original_bytes=len(raw),original_sha256=sha(raw),
                bytes=len(out),sha256=sha(out),transformations=transforms)
            count+=1
    prov['files']=list(records.values());write(base/'provenance.json',prov)
    withheld=json.loads((base/'withheld-manifest.json').read_text())
    bysha={v['original_sha256']:v['published'] for v in records.values()}
    for entry in withheld['files'].values():
        if entry.get('sha256') in bysha:
            entry['public_copy']=bysha[entry['sha256']];entry['availability']='public copy (see provenance)'
    write(base/'withheld-manifest.json',withheld)
    registry=json.loads((ROOT/'experiments/registry.json').read_text())
    for row in registry:
        if row['slug']=='canonical-context-size-generalization':
            row['published_files']=len(records);row['published_bytes']=sum(v['bytes'] for v in records.values())
    write(ROOT/'experiments/registry.json',registry)
    print(json.dumps({'verified_members_published':count,'source_archive_sha256':m['archive_sha256']}))

if __name__=='__main__':main()
