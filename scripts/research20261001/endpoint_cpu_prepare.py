"""Full CPU software preparation with explicit fixture labels; no CUDA work."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import endpoint_common as c
import endpoint_data as d


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    from endpoint_tests import cpu_tests,torch_cpu_tests,fake_parent
    from endpoint_fixture import run as fixture
    from endpoint_management import archive,verify,union
    import torch
    assert not torch.cuda.is_initialized()
    results=dict(data=cpu_tests(out/"data"),torch=torch_cpu_tests(out/"torch"))
    # Old coverage adapter must accept the new read-only Parent wrapper.
    c.import_old();import identification_data as old
    parent=fake_parent()
    for seed in c.SEEDS:
        for step in [1,32,4000,12000]:
            batch=old.training_batch(parent,seed,step,"I-F")
            assert batch["clean"].size==18432 and np.isfinite(batch["t"]).all()
    results["old_coverage_adapter"]="passed"
    # Full production bootstrap counts on synthetic data, to measure CPU cost.
    fixture(out/"software",full_reps=True)
    results["software"]=c.read(out/"software/fixture_complete.json")
    modeldata=out/"archive_fixture.npz";c.save(modeldata,array=np.arange(64))
    manifest=archive(c.ROOT,out/"package","archive_fixture",[modeldata,out/"data/checks.json"])
    verification=verify(out/"package/archive_fixture.tar.gz",out/"package/archive_fixture.manifest.json",out/"package/archive_fixture.verification.json")
    c.write(out/"science_manifest_fixture.json",dict(files=manifest["files"]))
    union(out/"science_manifest_fixture.json",[out/"package"],out/"package/union_fixture.json")
    results["archive"]=dict(status="passed",member_SHA_verified=verification["members"],union_paths=2)
    # Exercise post-exit closure with a fake output root, never the actual campaign root.
    import endpoint_management as management
    old_out,old_export=c.OUT,c.EXPORT
    try:
        c.OUT=out/"closure_fixture";c.EXPORT=out/"closure_package"
        c.write(c.OUT/"formal_started.json",dict(pid=99999999))
        c.write(c.OUT/"remote_export_complete.json",dict(fixture=True))
        c.write(c.OUT/"run_protocol.json",dict(sources={}))
        c.write(c.OUT/"status.json",dict(fixture=True))
        c.write(c.OUT/"science_manifest.json",dict(files=management.record_paths([modeldata])))
        closure=management.closure()
        verify(c.EXPORT/closure["archive"],c.EXPORT/"administrative_closure_v1.manifest.json",c.EXPORT/"administrative_closure_v1.verification.json")
        union(c.EXPORT/"closure_manifest_v1.json",[out/"package",c.EXPORT],c.EXPORT/"union_closure.json")
        results["administrative_closure_without_scientific_mutation"]="passed"
    finally:c.OUT,c.EXPORT=old_out,old_export
    assert not torch.cuda.is_initialized()
    result=dict(status="passed",cpu_only=True,cuda_initialized=False,time=time.time(),results=results,
                source_hashes={p.name:c.sha(p) for p in Path(__file__).parent.glob("*.py")})
    c.write(out/"checks.json",result)
    print(json.dumps(dict(status="passed",checks=str(out/"checks.json"),fixture=str(out/"software/fixture_complete.json"),gpu_budget_started=False)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
