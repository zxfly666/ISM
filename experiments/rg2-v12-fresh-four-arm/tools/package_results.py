"""Assemble a completed, audited RG-2 publication and separate large assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    workspace, snapshot, destination, assets = [p.resolve() for p in
            (args.workspace, args.snapshot, args.destination, args.assets)]
    results = workspace / 'results/rg2_v12_20261125'
    portable = workspace / 'tmp/rg2_run_20261125/project'
    banks = results / 'input_banks'
    assert not destination.exists() and not assets.exists(), 'Use new output directories'
    assert json.loads((snapshot/'snapshot_manifest.json').read_text())['stage'] == 'B_final'
    for stage in ('A', 'B'):
        assert json.loads((snapshot/f'runs/{stage}_independent_result_audit.json').read_text())['status'] == 'passed'
    assert json.loads((results/'stage_A/summary.json').read_text())['complete']
    assert json.loads((results/'stage_B/summary.json').read_text())['samples'] == 512
    copied = {}
    large = {'rg2-v12-checkpoints.tar.gz': [], 'rg2-v12-reproduction-inputs.tar.gz': []}

    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), relative
        shutil.copyfile(source, target)
        copied[relative] = dict(sha256=sha(target), bytes=target.stat().st_size)

    for folder in ('scripts', 'ism_diffusion', 'docs', 'output', 'runs'):
        for source in sorted((snapshot/folder).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            if source.name in ('ssh_transport.py', 'execution_scope.json'):
                continue
            relative = 'project/' + source.relative_to(snapshot).as_posix()
            if source.suffix == '.pt':
                large['rg2-v12-checkpoints.tar.gz'].append((source, relative))
            else:
                copy(source, relative)
    for source in sorted((portable/'inputs').rglob('*')):
        if not source.is_file():
            continue
        relative = 'project/' + source.relative_to(portable).as_posix()
        if source.suffix == '.pt':
            large['rg2-v12-reproduction-inputs.tar.gz'].append((source, relative))
        else:
            copy(source, relative)
    # Presentation-only revisions preserve the originally archived source too.
    for name in ('report.py','report_b.py','report_supplements.py'):
        local=workspace/'scripts/rg2_v12'/name
        relative='project/scripts/rg2_v12/'+name
        target=destination/relative
        if sha(local)!=sha(target):
            copy(target,'project/scripts/rg2_v12/history/'+name.replace('.py','_before_publication.py'))
            shutil.copyfile(local,target)
            copied[relative]=dict(sha256=sha(target),bytes=target.stat().st_size)
    copy(workspace/'scripts/rg2_v12/audit_statistics.py','project/scripts/rg2_v12/audit_statistics.py')
    for source in sorted((banks/'runs/A_supp/banks').glob('*.npz')):
        large['rg2-v12-reproduction-inputs.tar.gz'].append((source, 'project/'+source.relative_to(banks).as_posix()))
    train = workspace/'data/level1/parents_l1024.npz'
    assert sha(train) == '1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934'
    large['rg2-v12-reproduction-inputs.tar.gz'].append((train, 'project/data/level1/parents_l1024.npz'))
    for stage in ('stage_A', 'stage_B'):
        for source in sorted((results/stage).rglob('*')):
            if source.is_file():
                copy(source, stage+'/'+source.relative_to(results/stage).as_posix())
    # The historical loss panels can be regenerated without private asset paths.
    oldrel = Path('artifacts/rg_data_training_pilot_20261001_remote/extracted_rgpilot_v1/artifacts/rg_data_training_pilot_20261001')
    old = workspace/oldrel
    for source in [old/'science_manifest.json', *sorted((old/'training').glob('*/log.jsonl'))]:
        copy(source, 'project/'+source.relative_to(workspace).as_posix())
    for name in ('A0_archived_reproduction.json', 'stageA_diagnostics_verified.json',
                 'stageA_final_verified.json', 'stageB_final_verified.json', 'stageA_banks_verified.json',
                 'B_independent_statistics_audit.json'):
        copy(results/name, 'verification/'+name)
    copy(workspace/'output/RG-2_v1.2_advisor_outline.html', 'protocol/advisor_outline.html')
    copy(workspace/'scripts/rg2_v12/verify_snapshot.py', 'tools/verify_snapshot.py')
    copy(workspace/'scripts/rg2_v12/verify_publication.py', 'tools/verify_publication.py')
    copy(Path(__file__), 'tools/package_results.py')
    write(destination/'data-manifest.json', dict(files=copied, exact_bytes=True))

    assets.mkdir(parents=True)
    release = {}
    for name, members in large.items():
        archive = assets/name
        record = {relative: dict(sha256=sha(source), bytes=source.stat().st_size) for source, relative in members}
        manifest = assets/(name+'.manifest.json')
        write(manifest, dict(files=record, extract_relative_to='experiment directory'))
        with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
            for source, relative in members:
                tar.add(source, arcname=relative, recursive=False)
            tar.add(manifest, arcname=name+'.manifest.json', recursive=False)
        release[name] = dict(bytes=archive.stat().st_size, sha256=sha(archive), files=record)
    write(destination/'release-assets.json', release)
    write(assets/'release-assets.json', release)
    print(json.dumps(dict(public_files=len(copied), public_bytes=sum(v['bytes'] for v in copied.values()),
                          assets={k:dict(bytes=v['bytes'], sha256=v['sha256'], files=len(v['files']))
                                  for k,v in release.items()})), flush=True)


if __name__ == '__main__':
    main()
