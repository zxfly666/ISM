"""CPU fixtures only: no GPU, real reference generation or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(1,'D:/ISM/scripts/research20260921')
import numpy as np
import core_geometry_factorial_design as design
from core_geometry_factorial_data import training_batch, conditional_bank, array_hash


def main():
    random = np.random.default_rng(271828)
    parent = SimpleNamespace(lattice_size=1024,
        spins=(2*random.integers(0,2,(3,1024,1024),dtype=np.int8)-1),
        chain_ids=np.arange(3))
    widths_seen=set(); kinds_seen=set(); sparse_seen=set()
    for step in range(1,33):
        real = training_batch(parent,92501,step,'G11')
        summary = training_batch(parent,92501,step,'G11S')
        assert real['paired_data_hash'] == summary['paired_data_hash']
        assert real['paired_data_hash'] == training_batch(parent,92501,step,'G11')['paired_data_hash']
        if real['kind'] == 'continuous':
            assert real['actual_input_hash'] == summary['actual_input_hash']
        else:
            assert real['actual_input_hash'] != summary['actual_input_hash']
        ax=real['physical_axes']; o=real['origin']; p=real['parent']
        xx=(o[:,0,None]+ax[:,0])%1024; yy=(o[:,1,None]+ax[:,1])%1024
        truth=(parent.spins[p[:,None,None],xx[:,:,None],yy[:,None,:]]>0).astype(np.int64)
        truth[real['spin_flip']]=1-truth[real['spin_flip']]
        assert np.array_equal(truth,real['clean'])
        if real['sparse']:
            flat=real['noisy'].reshape(len(p),-1)
            flat_clean=real['clean'].reshape(len(p),-1)
            for row in range(len(p)):
                valid=real['query_valid'][row]; q=real['queries'][row,valid]
                assert len(q)==len(np.unique(q)) and np.all(flat[row,q]==2)
                assert np.array_equal(real['labels'][row,valid],flat_clean[row,q])
            assert real['query_valid'].sum()==512
        else:
            assert np.array_equal(real['noisy'],np.where(real['mask'],2,real['clean']))
        widths_seen.add(real['width']); kinds_seen.add(real['kind']); sparse_seen.add(real['sparse'])
    assert widths_seen==set(design.WIDTHS) and kinds_seen=={'continuous','train_gap'} and sparse_seen=={False,True}
    for arm in design.ARMS:
        row=training_batch(parent,92502,1,arm)
        assert row['clean'].size==18432
        if arm in ('G00','G01'): assert row['width']==48
        if arm in ('G00','G10'): assert row['kind']=='continuous'
    banks=[conditional_bank(parent,[0,1,2],2026092512,7,48,'held_gap',k) for k in (2,32,512)]
    for bank in banks[1:]:
        for key in ('clean','true_coords','summary_coords','queries','labels','parent','chain','origin'):
            assert np.array_equal(bank[key],banks[0][key])
        assert np.array_equal(bank['evidence'][:,:2],banks[0]['evidence'])
    fixture_p=np.array([.2,.8]); fixture_y=np.array([0.,1.])
    ce=-(fixture_y*np.log(fixture_p)+(1-fixture_y)*np.log(1-fixture_p)).mean()
    brier=((fixture_p-fixture_y)**2).mean()
    assert np.isclose(ce,-np.log(.8)) and np.isclose(brier,.04)
    result=dict(status='passed_cpu_data_adapter_only',
        tests=['paired_native_encodings','physical_sampling_matches_truth','deterministic_replay',
               'all_training_schedule_cells','512_total_auxiliary_slots','no_query_leakage',
               'ordinary_masks','fixed_factor_mapping','nested_evaluation_banks','CE_Brier_fixture'],
        data_source='synthetic_3_L1024_binary_fields',
        new_training_started=False,
        untested=['GPU_update','resume','bootstrap','full_runner','runtime_budget'],
        source_sha256=hashlib.sha256(Path(__file__).with_name('core_geometry_factorial_data.py').read_bytes()).hexdigest())
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
