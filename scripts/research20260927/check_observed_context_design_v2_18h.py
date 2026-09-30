"""Read-only standard-library validation of the 18h J design; no science."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
CONFIG = DOCS / "OBSERVED_CONTEXT_INTERVENTION_CONFIG_V2_18H_20260927.json"
PLAN = DOCS / "OBSERVED_CONTEXT_INTERVENTION_PLAN_V2_18H_20260927_ZH.md"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    spec = importlib.util.spec_from_file_location("j_v1_static", Path(__file__).with_name("check_observed_context_design.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        old.main()
    legacy = json.loads(capture.getvalue())
    saved = json.loads((ROOT / "artifacts/observed_context_intervention_20260927_design/static_design_checks_v1.json").read_text("utf-8"))
    for field in ["plan_sha256","config_sha256","checker_sha256"]:
        assert legacy[field] == saved[field], field
    c = json.loads(CONFIG.read_text("utf-8"))
    assert c["design_version"] == 2
    assert c["status"] == "design_only_not_implemented_not_preflighted_not_authorized_to_start"
    assert not c["authorized_to_start"] and not c["gpu_or_new_mc_started"] and not c["automation_created"]
    assert c["budget_started"] is None and c["deadline"] is None
    assert c["budget"]["hard_seconds"] == 64800 and c["budget"]["launch_gate_seconds"] == 63000
    cohorts = {x["id"]: x for x in c["cohorts"]}
    ct, fr = cohorts["continuation"], cohorts["fresh"]
    assert ct["updates"] == 4000 and fr["updates"] == 24000
    assert ct["base_step"] == 12000 and fr["base_step"] == 0
    assert not ct["fresh_initialization"] and fr["fresh_initialization"]
    assert set(fr["initialization_seeds"]).isdisjoint(ct["base_seeds"])
    assert set(fr["data_seeds"]).isdisjoint(ct["data_seeds"])
    assert len(fr["initialization_seeds"]) == len(ct["base_seeds"]) == 6
    assert c["primary"]["cohort"] == "fresh" and c["primary"]["left"] == "S-O_24000"
    assert not c["continuation_secondary"]["can_replace_failed_primary"]
    assert not c["primary"]["pooled_with_continuation"]
    assert math.isclose(c["retention"]["fresh"]["ci"], 1-.05/3)
    assert math.isclose(c["retention"]["continuation"]["ci"], 1-.05/6)
    assert c["arm_attention_modes"] == {"C-D":"dense","C-O":"observed_only","S-D":"dense","S-O":"observed_only"}
    tr = c["training_common"]
    nt, nm = 24, 30
    ni = len(fr["initialization_seeds"])*2*len(fr["formal_intermediate_core_only_steps"])
    parents = sum(x["parents"] for x in c["core_banks"])
    assert c["mc"]["chains"] * c["mc"]["parents_per_chain"] == 4096
    assert parents == 8704 and c["low_k"]["parents"] == 4096
    assert c["bootstrap"]["oracle_frames"] == c["exact_gibbs"]["formal"]["frames_per_chain"] == [0,64,128,192]
    x = {
      "trained_branches": nt, "continuation_branches": 12, "fresh_branches": 12,
      "frozen_baselines": 6, "final_evaluation_models": nm, "intermediate_core_models": ni,
      "continuation_training_updates": 12*ct["updates"], "fresh_training_updates": 12*fr["updates"],
      "training_updates": 12*(ct["updates"]+fr["updates"]), "fixture_updates": 2048,
      "continuation_windows_per_branch": ct["updates"]//32*4*2*sum(18432//w**2 for w in tr["widths"]),
      "fresh_windows_per_branch": fr["updates"]//32*4*2*sum(18432//w**2 for w in tr["widths"]),
      "continuation_tokens_per_branch": ct["updates"]*18432,
      "fresh_tokens_per_branch": fr["updates"]*18432,
      "total_training_tokens": 12*(ct["updates"]+fr["updates"])*18432,
      "continuation_sparse_slots_per_branch": ct["updates"]//32*8*512,
      "fresh_sparse_slots_per_branch": fr["updates"]//32*8*512,
      "mc_parents": 4096, "core_banks": 4,
      "core_final_prediction_files": nm*4, "core_intermediate_prediction_files": ni*4,
      "core_prediction_files": (nm+ni)*4, "core_input_forwards": (nm+ni)*parents,
      "mechanism_prediction_files": nm*8, "mechanism_input_forwards": nm*8*256,
      "padding_prediction_files": nm*6, "padding_input_forwards": nm*6*256,
      "oracle_prediction_files": nm*9, "oracle_input_forwards": nm*9*64*16,
      "low_k_prediction_files": nm, "low_k_input_forwards": nm*80*2*6,
      "validation_prediction_files": (12*len(ct["validation_steps"])+12*len(fr["validation_steps"]))*8,
      "validation_input_forwards": (12*len(ct["validation_steps"])+12*len(fr["validation_steps"]))*8*64,
      "recoverable_final_checkpoints": nt,
      "diagnostic_ema_checkpoints": 12*(len(ct["diagnostic_ema_steps"])+len(fr["diagnostic_ema_steps"])),
      "low_joint_count_entries": 4096*80*2*8, "figures": 10
    }
    modules = ["core","mechanism","padding","oracle","low_k","validation"]
    x["total_prediction_files"] = sum(x[m+"_prediction_files"] for m in modules)
    x["total_input_forwards_excluding_preflight"] = sum(x[m+"_input_forwards"] for m in modules)
    x["w96_input_forwards"] = (nm+ni)*4096 + nm*4*256 + nm*6*64*16
    x["other_input_forwards"] = x["total_input_forwards_excluding_preflight"]-x["w96_input_forwards"]
    assert x == c["expected_counts"], (x,c["expected_counts"])
    h = c["budget"]["historical_extrapolation"]
    rate48 = h["source_prediction_rate_seconds_by_width"]["W48"]
    rate96 = h["source_prediction_rate_seconds_by_width"]["W96"]
    T = x["training_updates"]/h["source_I_training_updates"]*h["source_I_training_hours"]*3600 + x["validation_input_forwards"]*rate48+h["assumption_checkpoint_IO_hours"]*3600
    M = h["source_I_reference_seconds"]*x["mc_parents"]/h["source_I_reference_parents"]
    F = x["w96_input_forwards"]*rate96+(x["other_input_forwards"]-x["validation_input_forwards"])*rate48
    L = h["source_I_prediction_io_seconds"]/h["source_I_conditional_input_forwards"]*x["total_input_forwards_excluding_preflight"]
    E = h["assumption_preflight_elapsed_hours"]*3600
    B = h["assumption_export_transfer_verify_hours"]*3600
    predicted = E+max(1.25*T,1.25*M)+1.25*(F+L)+2700+1.25*B+900+1800
    md = PLAN.read_text("utf-8")
    for link in re.findall(r"\]\((D:/[^)\n]+)\)", md):
        assert Path(re.sub(r":\d+$","",link)).exists(), link
    assert "17.17" in md and "827,520" in md and "1,656" in md
    print(json.dumps({
      "status":"passed_static_design_only_not_a_launch_gate",
      "config_sha256":sha(CONFIG), "plan_sha256":sha(PLAN), "checker_sha256":sha(__file__),
      "counts":x,
      "historical_timing_scenario":{"E_seconds":E,"T_seconds":T,"M_seconds":M,"F_seconds":F,"L_seconds":L,"B_seconds":B,
         "predicted_hours":predicted/3600,"new_mask_not_benchmarked":True,"launch_budget_not_verified":True},
      "v1_unchanged":True,"I_frozen_files_unchanged":legacy["unchanged_I_frozen_files"],
      "old_six_base_archived_final_hashes_verified":len(legacy["six_base_final_bytes_verified"]),
      "review_files_unchanged":len(legacy["unchanged_review_files"]),
      "not_performed":["GPU_preflight","training","new_MC","formal_evaluation","checkpoint_deserialization","independent_review_of_v2"],
      "authorized_to_start":False,"budget_started":None
    },ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()

