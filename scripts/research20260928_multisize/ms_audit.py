"""Read-only CPU audit of all paths, samples, labels, predictions and gates."""
from __future__ import annotations
import json
import math
import time
import numpy as np
import torch
import ms_common as c
import ms_data as d
import ms_model as m


def audit():
    started=time.time()
    proto=c.read(c.OUT/"run_protocol.json")
    c.check_sources(proto["sources"])
    parent=c.parent_verification()
    assert parent==proto["parent"]
    for path,digest in c.read(c.OUT/"input_freeze.json").items():
        assert c.sha(c.OUT/path)==digest
    bank=c.load(c.OUT/"banks/base.npz")
    original=c.load(c.PARENT/"bank.npz")
    assert bank.keys()==original.keys()
    for key in bank:
        np.testing.assert_array_equal(bank[key],original[key])
    views={s:c.load(c.OUT/"banks"/("k4_%d_fixed.npz"%s)) for s in (4,6)}
    lock=c.read(c.OUT/"all_models_locked.json")
    assert len(lock["finals"])==6
    log_rows=0
    tokens_by_arm={arm:0 for arm in c.CFG["arms"]}
    seconds_by_arm={arm:0. for arm in c.CFG["arms"]}
    final_records=[]
    for seed in c.CFG["seed_labels"]:
        pair={}
        initial=[]
        for arm in c.CFG["arms"]:
            folder=c.OUT/"training"/("s%d_%s"%(seed,arm))
            initial.append(c.read(folder/"initial.json"))
            with (folder/"train.jsonl").open(encoding="utf-8") as f:
                rows=[json.loads(line) for line in f]
            assert len(rows)==2048
            pair[arm]=rows
            seen=set()
            for step,row in enumerate(rows,1):
                c.deadline()
                assert row["step"]==step and row["seed"]==seed and row["arm"]==arm
                assert row["side"]==(6 if arm=="B46" and step%2 else 4)
                assert np.isfinite([row["loss"],row["grad_norm"],row["lr"],*row["family_losses"]]).all()
                assert row["loss"]>=0 and row["grad_norm"]>=0 and row["seconds"]>0
                assert row["lr"]==m.learning_rate(step)
                assert abs(np.mean(row["family_losses"])-row["loss"])<2e-7
                ids=d.indices(bank,seed,step)
                assert bank["train"][ids].all()
                seen.update(map(int,ids))
                first,second,physical,native=d.logical_and_native(bank,views,ids,row["side"])
                actual=[]
                for part in (first,second):
                    actual.extend([part[k].astype(np.float32) if k=="target" else part[k]
                                   for k in ("tokens","coordinates","t","query","target")])
                assert row["paired_data_digest"]==physical and row["native_full_precision_digest"]==native
                assert row["actual_input_digest"]==c.ahash(*actual)
                assert row["batch_row_digest"]==c.ahash(ids)
                assert row["microbatch_tokens"]==[512,16*row["side"]**2]
                tokens_by_arm[arm]+=sum(row["microbatch_tokens"])
                seconds_by_arm[arm]+=row["seconds"]
                log_rows+=1
            assert seen <= set(np.flatnonzero(bank["train"]))
            digest=c.sha(folder/"final.pt")
            assert digest==lock["finals"][folder.name]==c.read(folder/"complete.json")["final_sha256"]
            state,p=m.restore(folder/"final.pt")
            assert p["step"]==2048 and p["next_step"]==2049 and p["arm"]==arm and p["seed_label"]==seed
            assert p["protocol_hash"]==proto["protocol_hash"] and p["time"]<=lock["time"]
            for section in (p["model"],p["ema"]):
                assert all(torch.isfinite(v).all() for v in section.values())
            for row in p["optimizer"]["state"].values():
                assert int(row["step"])==2048 and all(torch.isfinite(v).all() for v in row.values() if torch.is_tensor(v))
            assert torch.equal(torch.get_rng_state(),p["torch_rng"])
            final_records.append(dict(cell=folder.name,final_sha256=digest,raw_hash=p["raw_hash"],ema_hash=p["ema_hash"],
                finite_model_EMA_AdamW=True,restored_identity_and_rng=True,distinct_train_rows_seen=len(seen)))
            del state,p
        assert initial[0]["model_hash"]==initial[1]["model_hash"]
        for a,b in zip(pair["A4"],pair["B46"]):
            assert a["paired_data_digest"]==b["paired_data_digest"]
            assert a["batch_row_digest"]==b["batch_row_digest"]
            assert (a["actual_input_digest"]==b["actual_input_digest"])==(a["step"]%2==0)
    banks={p.stem:(p,c.load(p)) for p in sorted((c.OUT/"banks").glob("*.npz"))}
    assert len(banks)==8
    for name,(_,b) in banks.items():
        if name!="base":
            d.validate_local(b)
    summary=c.read(c.OUT/"final_summary.json")
    rows=summary["model_rows"]
    counts=0
    primary={}
    for cell,digest in lock["finals"].items():
        for weight in ("raw","ema"):
            for name,(path,bank1) in banks.items():
                p=c.load(c.OUT/"evaluation"/cell/weight/(name+".npz"))
                assert float(p["started"])>=lock["time"]
                assert str(p["bank_sha256"])==c.sha(path)
                assert str(p["identity"])==cell+"_"+weight
                record=next(x for x in final_records if x["cell"]==cell)
                assert str(p["model_hash"])==record[weight+"_hash"]
                np.testing.assert_array_equal(p["target"],bank1["target"])
                np.testing.assert_array_equal(p["query"],bank1["query"])
                probability=p["probability"].astype(np.float64)
                y=bank1["target"].astype(np.float64)
                assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                clipped=np.clip(probability,1e-12,1-1e-12)
                kl=y*np.log(y/clipped)+(1-y)*np.log((1-y)/(1-clipped))
                np.testing.assert_allclose(p["kl"],kl,atol=2e-15,rtol=1e-9)
                np.testing.assert_allclose(p["error"],probability-y,atol=1e-15)
                np.testing.assert_allclose(p["brier"],(probability-y)**2+y*(1-y),atol=1e-15)
                seed=int(cell[1:6]);arm=cell[7:]
                if name=="base":
                    subsets={"k%d_%s"%(k,label):(bank1["k"]==k)&(bank1["train"]==train)
                             for k in (1,2,4) for label,train in (("train",True),("holdout",False))}
                    subsets={k:v for k,v in subsets.items() if v.any()}
                else:
                    subsets={name:np.ones(len(y),bool)}
                for label,sl in subsets.items():
                    row=next(x for x in rows if x["seed"]==seed and x["arm"]==arm and x["weight"]==weight and x["cell"]==label)
                    expected=c.summary(probability[sl],y[sl])
                    assert all(row[k]==v for k,v in expected.items())
                if name=="k4_12_fixed" and weight=="raw":
                    primary[(seed,arm)]=float(np.mean(kl))
                counts+=1
    diffs=[primary[(s,"B46")]-primary[(s,"A4")] for s in c.CFG["seed_labels"]]
    np.testing.assert_allclose(summary["primary"]["paired_mean_kl_B_minus_A"],diffs,atol=2e-15)
    assert summary["gates"]["paired_practical_improvement"]==all(v<=-.005 for v in diffs)
    assert counts==96 and len(rows)==144 and log_rows==12288
    for key,cells in (("B_new_size_ability",["k4_12_fixed"]),("B_retention",["k1_holdout","k2_holdout","k4_4_fixed"]),
                      ("B_trained_size_ability",["k4_6_fixed"])):
        sel=[r for r in rows if r["arm"]=="B46" and r["weight"]=="raw" and r["cell"] in cells]
        assert len(sel)==3*len(cells)
        assert summary["gates"][key]==all(r["max_kl"]<=.01 and r["max_probability_error"]<=.05 for r in sel)
    diagnostic=c.read(c.OUT/"diagnostic/summary.json")
    assert len(diagnostic["cells"])==78
    for row in diagnostic["cells"]:
        p=c.load(c.OUT/"diagnostic"/("seed_%d"%row["seed"])/(row["cell"]+".npz"))
        path=c.OUT/"diagnostic/banks"/(row["cell"]+".npz")
        bank1=c.load(path)
        assert str(p["bank_sha256"])==c.sha(path)
        assert float(p["finished"])<lock["time"]
        assert all(row[k]==v for k,v in c.summary(p["probability"],bank1["target"]).items())
    c.check_sources(proto["sources"])
    result=dict(status="passed",time=time.time(),seconds=time.time()-started,
        full_training_rows_rebuilt=log_rows,final_prediction_files_recomputed=counts,
        parent_diagnostic_prediction_files=78,finals=final_records,
        training_valid_tokens_by_arm=tokens_by_arm,training_update_seconds_by_arm=seconds_by_arm,
        all_physical_pairs_passed=True,all_six_finals_before_evaluation=True,
        old_sources_weights_and_data_unchanged=True,strict_primary_size_training_exclusion=True,
        exact_truth_no_new_MC=True,disk_free=c.disk())
    c.write(c.OUT/"audit.json",result)
    return result
