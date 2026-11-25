"""Verify frozen payloads and update only the publication-layer manifest.

Run after all reports, audits and figures are final. The original science/data
manifest is immutable. This script excludes ignored caches and its manifest
output from the publication inventory, and records exactly two editorial edits.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
PREFIX = ROOT.relative_to(REPO).as_posix() + '/'
BASELINE = ROOT / 'integrated/provenance/publication-manifest.baseline.json'
EDITS = {'REPORT_ZH.md', 'README.md'}

def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    baseline = json.loads(BASELINE.read_text(encoding='utf8'))['files']
    changed=[]
    for name, before in baseline.items():
        after=identity(ROOT/name)
        if after != before:
            assert name in EDITS, f'Unexpected original payload change: {name}'
            archived=name.replace('.md','.initial-20261125.md')
            assert identity(ROOT/archived)==before, f'Archive differs: {name}'
            changed.append({'path': name, 'before': before, 'after': after, 'exact_archive': archived})
    assert {x['path'] for x in changed}==EDITS
    provenance={
        'status':'passed', 'date':'2026-11-25',
        'science_commit':'5516370b3ae6e98160d9468668acdcbf851dc902',
        'historical_commit':'7e608a594c06ead3c484cdd5b38e1d8f2c4d56e3',
        'baseline_manifest':identity(BASELINE),
        'original_inventory_files':len(baseline),
        'original_files_byte_identical':len(baseline)-len(changed),
        'editorial_changes':changed,
        'no_original_scientific_payload_changed':True,
        'no_new_training_or_generation':True,
        'data_manifest_unchanged':identity(ROOT/'data-manifest.json')==baseline['data-manifest.json'],
        'authoritative_prose':'REPORT_ZH.md',
        'html_generated_from_same_markdown':json.loads((ROOT/'integrated/provenance/html_build.json').read_text(encoding='utf8'))['input_sha256']==identity(ROOT/'REPORT_ZH.md')['sha256'],
        'scope':'Only README and REPORT are replaced; exact originals archived at their original relative-link depth. All other additions are derived report materials, auditing code or offline reading assets.',
        'excluded_from_manifest':['publication-manifest.json','git-ignored caches'],
        'publication_script_sha256':identity(Path(__file__))['sha256']
    }
    assert provenance['data_manifest_unchanged'] and provenance['html_generated_from_same_markdown']
    record=ROOT/'integrated/provenance/editorial_provenance.json'
    record.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    out=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',PREFIX],cwd=REPO)
    names={x[len(PREFIX):] for x in out.decode('utf8').splitlines() if x.startswith(PREFIX)}
    names.update(baseline)
    names.discard('publication-manifest.json')
    assert 'integrated/provenance/editorial_provenance.json' in names
    files={n:identity(ROOT/n) for n in sorted(names)}
    result={'schema':'rg2-publication-manifest-v2','science_commit':provenance['science_commit'],
            'editorial_revision':'integrated-report-20261125',
            'editorial_provenance':'integrated/provenance/editorial_provenance.json','files':files}
    (ROOT/'publication-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'status':'passed','original_files':len(baseline),'byte_identical':len(baseline)-len(changed),'editorial_edits':sorted(EDITS),'publication_files':len(files),'publication_bytes':sum(v['bytes'] for v in files.values())}))

if __name__=='__main__': main()
