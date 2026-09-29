"""Explicit member-hashed bundle; never overwrites an existing different file."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import sg_common as c
import sg_management as manager


def package():
    paths=[c.ROOT/p for p in c.source_snapshot()]
    paths += [p for p in (c.OUT/'cpu_preparation').rglob('*') if p.is_file() and p.suffix!='.pt' and '__pycache__' not in p.parts]
    paths += [c.OUT/'preflight_visual_review.json']
    return manager.archive(c.ROOT,c.EXPORT,'deploy_v1',paths)


def install(archive,manifest,root):
    root=Path(root).resolve();data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    assert c.sha(archive)==data['archive_sha256']
    expected={r['path']:r for r in data['files']};seen=set();created=0;matched=0
    with tarfile.open(archive,'r:gz') as tar:
        for entry in tar:
            assert entry.isfile() and entry.name in expected and entry.name not in seen
            path=(root/entry.name).resolve();assert path.is_relative_to(root) and path!=root
            assert '..' not in Path(entry.name).parts
            with tar.extractfile(entry) as f:content=f.read()
            assert len(content)==expected[entry.name]['bytes'] and hashlib.sha256(content).hexdigest()==expected[entry.name]['sha256']
            if path.exists():
                assert path.is_file() and c.sha(path)==expected[entry.name]['sha256'],str(path)
                matched+=1
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(content)
                created+=1
            seen.add(entry.name)
    assert seen==set(expected)
    return dict(status='passed',created=created,existing_identical=matched,members=len(seen))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('package','install'),required=True)
    p.add_argument('--archive');p.add_argument('--manifest');p.add_argument('--root')
    a=p.parse_args();print(json.dumps(package() if a.mode=='package' else install(a.archive,a.manifest,a.root)))
