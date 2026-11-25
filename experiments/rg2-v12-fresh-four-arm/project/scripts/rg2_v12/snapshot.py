"""Archive completed experimental outputs without copying historical data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['A_diagnostics', 'A_final', 'B_final'], required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs = root / 'runs'
    if args.stage == 'A_diagnostics':
        assert json.loads((runs/'campaign/exact_blocks.json').read_text())['status'] == 'complete'
    else:
        assert json.loads((runs/'campaign/status.json').read_text())['status'] == 'stage_A_computation_complete_review_required'
    if args.stage == 'B_final':
        assert json.loads((runs/'campaign_B/status.json').read_text())['status'] == 'B_complete_review_required'
    selected = ['runs/A_v2', 'runs/A_supp', 'runs/campaign', 'scripts', 'ism_diffusion', 'docs', 'output']
    if args.stage == 'B_final':
        selected += ['runs/B_v2', 'runs/B_eval', 'runs/campaign_B']
    paths = []
    for folder in selected:
        paths += [p for p in (root/folder).rglob('*') if p.is_file()]
    paths += [p for p in runs.glob('*.json')]
    paths += [root/'inputs/manifest.json']
    paths = sorted(set(p for p in paths if '__pycache__' not in p.parts
                       and 'banks' not in p.relative_to(root).parts
                       and p.suffix not in ('.lock', '.tmp', '.pid')
                       and not (args.stage == 'A_diagnostics' and
                                (any(k in p.parts for k in ('samples', 'metrics')) or p.name == 'generation.log'))))
    assert not args.archive.exists(), 'Use a new snapshot filename'
    manifest = dict(stage=args.stage, created_unix=time.time(), root=str(root),
                    files={p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths},
                    exclusions=['historical training data; available at recorded source path',
                                'evaluation input banks; preserved remotely and in input manifests',
                                'temporary/lock/pid/bytecode files'])
    if args.stage == 'A_diagnostics':
        manifest['exclusions'].append('generation samples/metrics are still running')
    info = args.archive.with_suffix('.manifest.json')
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    with tarfile.open(args.archive, 'w:gz', compresslevel=3) as tar:
        for p in paths:
            tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        tar.add(info, arcname='snapshot_manifest.json', recursive=False)
    result = dict(archive=str(args.archive),sha256=sha(args.archive),bytes=args.archive.stat().st_size,
                  files=len(paths),manifest=str(info))
    args.archive.with_suffix('.receipt.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
