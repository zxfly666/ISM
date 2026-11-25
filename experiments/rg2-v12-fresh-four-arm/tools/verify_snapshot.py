"""Verify and unpack a completed snapshot, refusing path escapes and overwrites."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf8'))
    digest = sha(args.archive)
    assert digest == receipt['sha256'], 'Archive SHA-256 mismatch'
    assert args.archive.stat().st_size == receipt['bytes']
    assert not args.out.exists(), 'Use a new extraction directory'
    root = args.out.resolve()
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert len(set(names)) == len(names), 'Duplicate archive paths'
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute()
            assert '..' not in path.parts and '\\' not in member.name and ':' not in member.name
            assert (root / member.name).resolve().is_relative_to(root)
        manifest = json.load(archive.extractfile('snapshot_manifest.json'))
        assert set(names) == set(manifest['files']) | {'snapshot_manifest.json'}
        assert len(manifest['files']) == receipt['files']
        root.mkdir(parents=True)
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            if member.name == 'snapshot_manifest.json':
                continue
            expected = manifest['files'][member.name]
            assert target.stat().st_size == expected['bytes'], member.name
            assert sha(target) == expected['sha256'], member.name
    result = dict(status='passed', archive_sha256=digest, archive_bytes=receipt['bytes'],
                  verified_files=len(manifest['files']), stage=manifest['stage'],
                  archive=str(args.archive.resolve()), extracted=str(root))
    args.verification.parent.mkdir(parents=True, exist_ok=True)
    args.verification.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
