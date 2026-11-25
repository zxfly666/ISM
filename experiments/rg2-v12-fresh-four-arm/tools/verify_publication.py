"""Check the public file manifest and optional large assets; safely extract them."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--assets',type=Path);p.add_argument('--extract',action='store_true');args=p.parse_args()
    root=args.root.resolve();assert not args.extract or args.assets
    manifest=json.loads((root/'publication-manifest.json').read_text(encoding='utf8'))
    for name,expected in manifest['files'].items():
        path=root/name;assert path.resolve().is_relative_to(root)
        assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],name
    count=0
    if args.assets:
        assets=json.loads((root/'release-assets.json').read_text(encoding='utf8'))
        for name,record in assets.items():
            archive=args.assets/name
            assert archive.stat().st_size==record['bytes'] and sha(archive)==record['sha256'],name
            with tarfile.open(archive,'r:gz') as tar:
                members=tar.getmembers();names=[m.name for m in members]
                assert len(set(names))==len(names)
                assert set(names)==set(record['files'])|{name+'.manifest.json'}
                for member in members:
                    part=PurePosixPath(member.name);target=(root/member.name).resolve()
                    assert member.isfile() and not part.is_absolute() and '..' not in part.parts
                    assert '\\' not in member.name and ':' not in member.name and target.is_relative_to(root)
                    if member.name==name+'.manifest.json':continue
                    expected=record['files'][member.name];h=hashlib.sha256();size=0
                    with tar.extractfile(member) as source:
                        for block in iter(lambda:source.read(1<<20),b''):h.update(block);size+=len(block)
                    assert size==expected['bytes'] and h.hexdigest()==expected['sha256'],member.name
                    if args.extract:
                        if target.exists():
                            assert target.stat().st_size==size and sha(target)==expected['sha256'],member.name
                        else:
                            target.parent.mkdir(parents=True,exist_ok=True)
                            with tar.extractfile(member) as source,target.open('xb') as destination:
                                for block in iter(lambda:source.read(1<<20),b''):destination.write(block)
                    count+=1
    print(json.dumps(dict(status='passed',public_files=len(manifest['files']),asset_members=count,extracted=args.extract)),flush=True)


if __name__=='__main__':main()
