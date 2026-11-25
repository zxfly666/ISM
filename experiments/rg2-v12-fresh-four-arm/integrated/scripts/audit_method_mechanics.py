"""CPU-only checks of documented operators, pairing and nested reveal schedule.

Executes only the frozen NumPy batching functions extracted with AST; no model,
GPU, training or generation is run. Synthetic fields test implementation rules.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


class Coordinates:
    def __init__(self,value):self.value=value
    def numpy(self):return self.value


def coordinates(width,level,device):
    axis=np.arange(width,dtype=np.float32)
    if level:axis=3*axis+1
    row,col=np.meshgrid(axis,axis,indexing='ij')
    return Coordinates(np.stack((row,col),axis=-1))


def extract(path,functions,assignments=()):
    tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in assignments for target in node.targets):nodes.append(node)
    return ast.Module(body=nodes,type_ignores=[])


def main():
    source=ROOT/'project/scripts/rg2_v12/phase_b.py'
    environment={'np':np,'hashlib':hashlib,'TRAIN_SHA':'not-used-by-batch','a':SimpleNamespace(coords=coordinates)}
    exec(compile(extract(source,('rand','digest','batch'),('CONFIG',)),str(source),'exec'),environment)
    fields=np.random.default_rng(118).integers(0,2,(3,1024,1024),dtype=np.int8)
    checked=0;spot_checks=0;role_selections=[]
    for seed in (112501,112502):
        for cycle in range(1,5):
            width=[16,24,32,48][cycle-1];count=18432//width**2
            roles={}
            for role in ('common','aux'):
                rows={arm:environment['batch'](fields,seed,cycle,role,arm) for arm in ('F','P','D','M')}
                assert len({x['selection_hash'] for x in rows.values()})==1
                assert len({x['noise_hash'] for x in rows.values()})==1
                assert rows['F']['spin_hash']==rows['P']['spin_hash']
                assert len({rows[k]['coordinate_hash'] for k in ('P','D','M')})==1
                if role=='common':assert len({x['input_hash'] for x in rows.values()})==1
                for arm,batch in rows.items():
                    assert batch['noisy'].shape==(count,width,width)
                    assert batch['coords'].shape==(count,width,width,2)
                    assert batch['t'].shape==(count,) and batch['mask'].dtype==bool
                    assert np.array_equal(batch['noisy']==2,batch['mask'])
                    assert batch['mask'][batch['endpoint']].all()
                    # Check literal physical parent sites and blocks, including
                    # random periodic origins, independently of reshape syntax.
                    for b in range(count):
                        p=int(batch['parent'][b]);oy,ox=map(int,batch['origin'][b])
                        for i,j in ((0,0),(width-1,width-1),(width//2,width//2)):
                            if role=='common':expected=fields[p,(oy+i)%1024,(ox+j)%1024]
                            elif arm in ('F','P'):expected=fields[p,(oy+width+i)%1024,(ox+width+j)%1024]
                            elif arm=='D':expected=fields[p,(oy+3*i+1)%1024,(ox+3*j+1)%1024]
                            else:expected=int(sum(int(fields[p,(oy+3*i+di)%1024,(ox+3*j+dj)%1024]) for di in range(3) for dj in range(3))>=5)
                            if batch['flip'][b]:expected=1-expected
                            assert batch['clean'][b,i,j]==expected
                            spot_checks+=1
                    base=coordinates(width,int(role=='aux' and arm!='F'),'cpu').numpy()
                    for index,code in enumerate(batch['d4']):
                        matrix=np.eye(2,dtype=int)
                        if code&1:matrix=matrix[:,::-1]
                        if code&2:matrix[:,0]*=-1
                        if code&4:matrix[:,1]*=-1
                        assert np.array_equal(batch['coords'][index],base@matrix)
                    checked+=1
                roles[role]=rows['F']['selection_hash']
            role_selections.append(roles['common']!=roles['aux'])

    sampling=ROOT/'project/scripts/rg2_v12/phase_a.py'
    env={'np':np}
    exec(compile(extract(sampling,('schedule','reveal_at','uniform_fields'),('STEPS','RNG_ROOT')),str(sampling),'exec'),env)
    reveal_checks=0
    for width in (32,96):
        for seed in (112501,112502):
            for sample in range(4):
                u,v=env['uniform_fields'](seed,width,sample)
                j64,j256=env['reveal_at'](u,64),env['reveal_at'](u,256)
                assert np.array_equal(j64,j256//4)
                assert ((u>0)&(u<1)&(v>0)&(v<1)).all()
                for s,idx in ((64,j64),(256,j256)):
                    t=env['schedule'](s)
                    assert ((t[idx]>=u)&(u>t[idx+1])).all()
                    assert (idx>=0).all() and (idx<s).all()
                reveal_checks+=1
    parameter_total=512+131712+7*(2*33024+65664+131712)+514+256
    assert parameter_total==1976962
    record=dict(status='passed',scope='synthetic CPU input operators and scheduled reveal identities only; no trained model executed',
                batch_cases=checked,literal_parent_site_checks=spot_checks,
                coordinate_D4_matrix_equivalence=True,common_aux_selections_differ_in_tested_cases=all(role_selections),
                paired_reveal_cases=reveal_checks,derived_parameter_total=parameter_total,
                source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,sampling)},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'integrated/provenance/method_mechanics_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':main()
