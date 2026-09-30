"""All diagnostic and final cells are retained; no model selection."""
from __future__ import annotations
import math
import time
import numpy as np
import ms_common as c
import ms_data as d
import ms_model as m


def score(path, model, bank, bank_path, identity):
    started = time.time()
    p = m.predict(model, bank)
    values = c.metrics(p, bank["target"])
    c.save(path, probability=p, target=bank["target"], query=bank["query"],
           **values, model_hash=np.array(m.model_hash(model)),
           bank_sha256=np.array(c.sha(bank_path)), identity=np.array(identity),
           started=np.array(started), finished=np.array(time.time()))
    return p


def factor_diagnostics(parent_record):
    banks, meta = d.diagnostic_banks()
    all_rows, effects = [], []
    folder = c.OUT/"diagnostic"
    c.write(folder/"design.json", meta)
    for name, bank in banks.items():
        c.save(folder/"banks"/(name+".npz"), **bank)
    for seed in c.CFG["parent_seeds"]:
        model = m.old_model(seed, parent_record["old_final_sha256"][str(seed)])
        predictions = {}
        for name, bank in banks.items():
            p = score(folder/("seed_%d"%seed)/(name+".npz"), model, bank,
                      folder/"banks"/(name+".npz"), "parent_raw_%d"%seed)
            predictions[name] = p
            all_rows.append(dict(seed=seed, cell=name, **c.summary(p,bank["target"])))
        for order in ("near","far"):
            pad = float(np.max(np.abs(predictions["pad64_from16_"+order]-predictions["n16_r1_t075_"+order])))
            assert pad <= c.CFG["strict_probability_tolerance"]
            effects.append(dict(seed=seed, factor="PAD", order=order, max_probability_change=pad))
            # Full conditional effects, not a single additive causal decomposition.
            for n in (16,32,64):
                for r in (1,2):
                    a,b = "n%d_r%d_t075_%s"%(n,r,order),"n%d_r%d_t09375_%s"%(n,r,order)
                    effects.append(effect(seed,"clock",a,b,predictions,banks))
                for t in ("075","09375"):
                    a,b = "n%d_r1_t%s_%s"%(n,t,order),"n%d_r2_t%s_%s"%(n,t,order)
                    effects.append(effect(seed,"peripheral_coordinates",a,b,predictions,banks))
            for r in (1,2):
                for t in ("075","09375"):
                    for n in (32,64):
                        a,b = "n16_r%d_t%s_%s"%(r,t,order),"n%d_r%d_t%s_%s"%(n,r,t,order)
                        effects.append(effect(seed,"valid_token_count_and_nested_set",a,b,predictions,banks))
        del model
    result = dict(status="completed", time=time.time(), source="three unchanged parent raw models",
                  cells=all_rows, effects=effects, new_primary_size_evaluated=False,
                  N64_layouts_same_set_not_independent=True,
                  causal_scope="controlled representation changes, not unique internal mechanism identification")
    c.write(folder/"summary.json",result)
    return result


def effect(seed,factor,a,b,pred,banks):
    kl_a = c.metrics(pred[a],banks[a]["target"])["kl"]
    kl_b = c.metrics(pred[b],banks[b]["target"])["kl"]
    return dict(seed=seed,factor=factor,from_cell=a,to_cell=b,
        mean_kl_change=float((kl_b-kl_a).mean()),
        mean_absolute_probability_change=float(np.abs(pred[b]-pred[a]).mean()),
        max_probability_change=float(np.abs(pred[b]-pred[a]).max()))


def final_evaluation():
    lock = c.read(c.OUT/"all_models_locked.json")
    assert len(lock["finals"]) == 6
    for cell, digest in lock["finals"].items():
        assert c.sha(c.OUT/"training"/cell/"final.pt") == digest
    base = c.load(c.OUT/"banks/base.npz")
    # Held-size inputs first materialized for model evaluation after all six finals lock.
    for natural in (False,True):
        key = "k4_12_"+("natural" if natural else "fixed")
        c.save(c.OUT/"banks"/(key+".npz"),**d.extend_k4(base,12,natural))
    paths = {p.stem:p for p in sorted((c.OUT/"banks").glob("*.npz"))}
    rows = []
    for cell in lock["finals"]:
        state, payload = m.restore(c.OUT/"training"/cell/"final.pt")
        assert payload["step"] == c.CFG["steps"]
        seed,arm = payload["seed_label"],payload["arm"]
        for weight,model in zip(("raw","ema"),state[:2]):
            for key,path in paths.items():
                bank = c.load(path)
                p = score(c.OUT/"evaluation"/cell/weight/(key+".npz"),model,bank,path,cell+"_"+weight)
                if key == "base":
                    for k in (1,2,4):
                        for split,train in (("train",True),("holdout",False)):
                            select = (bank["k"] == k) & (bank["train"] == train)
                            if not select.any():
                                continue
                            rows.append(dict(seed=seed,arm=arm,weight=weight,cell="k%d_%s"%(k,split),
                                             **c.summary(p[select],bank["target"][select])))
                else:
                    rows.append(dict(seed=seed,arm=arm,weight=weight,cell=key,**c.summary(p,bank["target"])))
        c.write(c.OUT/"evaluation"/cell/"complete.json",dict(status="complete",time=time.time(),final_sha256=lock["finals"][cell]))
        del state
    contrasts = []
    for weight in ("raw","ema"):
        for cell in sorted(set(r["cell"] for r in rows)):
            diffs=[]
            for seed in c.CFG["seed_labels"]:
                pair={r["arm"]:r for r in rows if r["seed"]==seed and r["weight"]==weight and r["cell"]==cell}
                diffs.append(pair["B46"]["mean_kl"]-pair["A4"]["mean_kl"])
            contrasts.append(dict(cell=cell,weight=weight,seed_labels=c.CFG["seed_labels"],paired_mean_kl_B_minus_A=diffs,
                mean=float(np.mean(diffs)),seed_sd=float(np.std(diffs,ddof=1)),paired_mcse=float(np.std(diffs,ddof=1)/math.sqrt(3))))
    primary = next(r for r in contrasts if r["cell"]=="k4_12_fixed" and r["weight"]=="raw")
    def allpass(cells):
        selected=[r for r in rows if r["weight"]=="raw" and r["arm"]=="B46" and r["cell"] in cells]
        assert len(selected)==3*len(cells)
        return all(r["ability_passed"] for r in selected)
    gates=dict(B_new_size_ability=allpass(["k4_12_fixed"]),
        B_retention=allpass(["k1_holdout","k2_holdout","k4_4_fixed"]),
        B_trained_size_ability=allpass(["k4_6_fixed"]),
        paired_practical_improvement=all(x <= c.CFG["paired_mean_kl_margin"] for x in primary["paired_mean_kl_B_minus_A"]))
    gates["all_required_diagnostic_gates"] = all(gates.values())
    baselines={}
    for key,path in paths.items():
        b=c.load(path)
        baselines[key]={"half":c.summary(np.full(len(b["target"]),.5),b["target"])}
        if key != "base":
            bits=np.sum(np.where(b["tokens"]<2,2*b["tokens"]-1,0),(1,2))
            oracle=1/(1+np.exp(-2*c.BETA*bits))
            baselines[key]["four_neighbour_counting_oracle"]=c.summary(oracle,b["target"])
    result=dict(status="science_complete_pending_audit_backup",time=time.time(),model_rows=rows,
        paired_contrasts=contrasts,primary=primary,gates=gates,baselines=baselines,
        weights_primary="final_raw",reference_MCSE=0,
        inference_scope="three paired initialization seeds; exact finite tasks; not population confirmation",
        natural_clock_secondary=True,EMA_secondary=True,new_architecture=False,new_MC=False)
    c.write(c.OUT/"final_summary.json",result)
    return result
