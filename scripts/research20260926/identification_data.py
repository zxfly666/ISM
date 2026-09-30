"""CPU data: common physical stream, native F/S/R, no old-module mutation."""
import numpy as np
import identification_common as c

def increments(w,kind):
    q=w//4-1
    if kind=='continuous':return np.ones(w-1,dtype=np.int64)
    if kind=='train_gap':return np.repeat([1,2,4,8],[q+1,q,q+2,q])
    if kind in ['held_gap','clustered_held_gap']:return np.repeat([3,6],[3*q+3,q])
    raise ValueError(kind)

def axes(w,kind,batch,seed,index,role='physical_axes'):
    random=c.rng(seed,index,role);inc=increments(w,kind)
    out=np.zeros((batch,2,w),dtype=np.int64)
    for b in range(batch):
        for a in range(2):
            g=np.roll(inc,int(random.integers(len(inc)))) if kind=='clustered_held_gap' and role=='physical_axes' else random.permutation(inc)
            out[b,a,1:]=g.cumsum()
    assert out.max()<512
    return out

def coordinates(a,arm,d4=None):
    a=np.asarray(a,dtype=np.float32).copy();b,_,w=a.shape
    if arm=='I-S':
        span=a[:,:,-1:].copy();a=span/(w-1)*np.arange(w,dtype=np.float32);a[:,:,-1:]=span
    elif arm not in ('I-F','I-R'):raise ValueError(arm)
    z=np.empty((b,w,w,2),dtype=np.float32)
    z[...,0]=a[:,0,:,None];z[...,1]=a[:,1,None,:]
    if d4 is not None:
        for i,code in enumerate(d4):
            if int(code)&1:z[i]=z[i][...,::-1].copy()
            if int(code)&2:z[i,...,0]*=-1
            if int(code)&4:z[i,...,1]*=-1
    return z

def schedule(seed,step):
    if not 1<=step<=c.STEPS:raise ValueError(step)
    cycle,offset=divmod(step-1,32)
    cell,repeat=divmod(int(c.rng(seed,cycle,'balanced_schedule').permutation(32)[offset]),4)
    w=c.WIDTHS[cell%4]
    return dict(width=w,kind='continuous' if cell<4 else 'train_gap',sparse=repeat==3,batch=c.TOKENS//w**2,latent_cell=cell)

def sampled(parent,ids,a,origins):
    if parent.lattice_size!=1024:raise ValueError('L1024 required')
    x=(origins[:,0,None]+a[:,0])%1024;y=(origins[:,1,None]+a[:,1])%1024
    spins=parent.spins[ids[:,None,None],x[:,:,None],y[:,None,:]]
    if not np.isin(spins,[-1,1]).all():raise ValueError('Invalid reference spins')
    return (spins>0).astype(np.int8)

def training_batch(parent,seed,step,arm):
    sc=schedule(seed,step);b=sc['batch'];w=sc['width'];n=w*w;kind=sc['kind']
    a=axes(w,kind,b,seed,step)
    ids=c.rng(seed,step,'parent').integers(len(parent.spins),size=b)
    origins=c.rng(seed,step,'origin').integers(1024,size=(b,2))
    clean=sampled(parent,ids,a,origins).astype(np.int64)
    flip=c.rng(seed,step,'spin_flip').random(b)<.5;clean[flip]=1-clean[flip]
    d4=c.rng(seed,step,'d4').integers(8,size=b)
    native=axes(w,kind,b,seed,step,'independent_random_coordinates') if arm=='I-R' else a
    coords=coordinates(native,arm,d4)
    if sc['sparse']:
        k=c.rng(seed,step,'sparse_k').choice(c.CONFIG['sparse_training_ks'],size=b)
        count,rem=divmod(512,b);counts=np.full(b,count);counts[c.rng(seed,step,'query_allocation').permutation(b)[:rem]]+=1
        noisy=np.full((b,n),2,dtype=np.int64);query=np.zeros((b,64),dtype=np.int64);valid=np.zeros((b,64),dtype=bool);labels=np.zeros((b,64),dtype=np.int64)
        random=c.rng(seed,step,'query_and_observations');flat=clean.reshape(b,n)
        for i in range(b):
            order=random.permutation(n);q=order[:counts[i]];e=order[counts[i]:counts[i]+k[i]]
            query[i,:counts[i]]=q;valid[i,:counts[i]]=True;labels[i,:counts[i]]=flat[i,q];noisy[i,e]=flat[i,e]
        noisy=noisy.reshape(b,w,w);mask=noisy==2;t=(1-k/n).astype(np.float32)
        assert valid.sum()==512
    else:
        t=c.rng(seed,step,'time').uniform(.01,1,b).astype(np.float32)
        t[c.rng(seed,step,'endpoint').random(b)<.02]=1
        mask=c.rng(seed,step,'mask').random(clean.shape)<t[:,None,None]
        noisy=np.where(mask,2,clean);k=(~mask).sum((1,2))
        query=np.empty((b,0),dtype=np.int64);valid=np.empty((b,0),dtype=bool);labels=np.empty((b,0),dtype=np.int64)
    paired=c.ah(clean,a,ids,origins,flip,d4,noisy,t,query,valid,labels)
    return dict(**sc,clean=clean,physical_axes=a,coords=coords,parent=ids,origin=origins,d4=d4,spin_flip=flip,
                noisy=noisy,mask=mask,t=t,k=k,queries=query,query_valid=valid,labels=labels,
                paired_data_hash=paired,actual_input_hash=c.ah(noisy,coords,t))

def conditional_bank(parent,ids,seed,geometry_index,w,kind,k):
    ids=np.asarray(ids,dtype=np.int64);n=len(ids);a=axes(w,kind,n,seed,geometry_index)
    origins=c.rng(seed,geometry_index,'bank_origin').integers(1024,size=(n,2))
    clean=sampled(parent,ids,a,origins);flat=clean.reshape(n,-1)
    random=c.rng(seed,geometry_index,'bank_query_and_evidence')
    q=np.empty((n,64),dtype=np.int64);e=np.empty((n,k),dtype=np.int64);noisy=np.full(flat.shape,2,dtype=np.int8)
    for i in range(n):
        order=random.permutation(w*w);q[i]=order[:64];e[i]=order[64:64+k];noisy[i,e[i]]=flat[i,e[i]]
    label=flat[np.arange(n)[:,None],q]
    return dict(clean=clean,physical_axes=a,origin=origins,parent=ids,chain=np.asarray(parent.chain_ids)[ids],
                noisy=noisy.reshape(n,w,w),queries=q,evidence=e,labels=label,t=np.full(n,1-k/(w*w),dtype=np.float32),
                width=np.array(w),kind=np.array(kind),k=np.array(k),geometry_index=np.array(geometry_index),bank_seed=np.array(seed))

def native_coordinates(bank,arm,rep=0):
    if 'input_coordinates' in bank:return bank['input_coordinates']
    a=bank['physical_axes'];kind=str(bank['kind']);w=int(bank['width'])
    if arm=='I-R' and kind!='continuous':
        a=axes(w,kind,len(a),int(bank['bank_seed']),int(bank['geometry_index']),f'bank_random_coordinates_{rep}')
    return coordinates(a,arm)

def common_hash(bank):return c.ah(bank['noisy'],bank['physical_axes'],bank['t'],bank['queries'],bank['labels'],bank['parent'],bank['chain'])

def validate_bank(bank):
    b=len(bank['parent']);n=int(bank['width'])**2;rows=np.arange(b)[:,None]
    q=bank['queries'];e=bank['evidence'];noisy=bank['noisy'].reshape(b,n);flat=bank['clean'].reshape(b,n)
    assert np.all(noisy[rows,q]==2) and np.array_equal(flat[rows,q],bank['labels'])
    assert np.array_equal(noisy[rows,e],flat[rows,e])
    assert all(len(np.unique(np.r_[q[i],e[i]]))==64+int(bank['k']) for i in range(b))
    assert np.all((noisy!=2).sum(1)==int(bank['k']))

def padding_banks(parent,ids,k,seed=2026092621):
    b=conditional_bank(parent,ids,seed,80,48,'continuous',k)
    small={**b,'input_coordinates':coordinates(b['physical_axes'],'I-F')+24}
    large=np.full((len(ids),96,96),2,dtype=np.int8);large[:,24:72,24:72]=b['noisy']
    q=(b['queries']//48+24)*96+(b['queries']%48+24)
    large_axis=np.broadcast_to(np.arange(96),(len(ids),2,96)).copy()
    fixed={**b,'width':np.array(96),'noisy':large,'queries':q,'input_coordinates':coordinates(large_axis,'I-F')}
    natural={**fixed,'t':np.full(len(ids),1-k/9216,dtype=np.float32)}
    assert np.all(large.reshape(len(ids),-1)[np.arange(len(ids))[:,None],q]==2)
    assert np.array_equal(small['input_coordinates'],fixed['input_coordinates'][:,24:72,24:72])
    return [small,fixed,natural]

def low_layouts():
    result=[];types=[]
    for kind,n,negative in [('train_gap',32,False),('held_gap',32,False),('train_gap',8,True),('held_gap',8,True)]:
        for _ in range(n):
            idx=len(result);random=c.rng(c.CONFIG['role_seeds']['low_k'],idx,'low_layout')
            a=axes(48,kind,1,c.CONFIG['role_seeds']['low_k'],idx)[0]
            near,far=(1,8) if kind=='train_gap' else (3,6)
            rest=increments(48,kind).tolist();rest.remove(near);rest.remove(far)
            slots=[2,3] if negative else [22,23]
            gap=np.empty(47,dtype=np.int64);gap[slots]=[near,far]
            gap[[i for i in range(47) if i not in slots]]=random.permutation(rest)
            a[1]=np.r_[0,gap.cumsum()];alt=a.copy();gap[slots]=gap[slots[::-1]];alt[1]=np.r_[0,gap.cumsum()]
            result.append(np.stack([a,alt]));types.append(kind+('_negative' if negative else '_main'))
    return dict(axes=np.array(result),types=np.array(types),query=np.array([23,22]),visible=np.array([[23,23],[23,25]]))

def low_inputs(layouts,arm,rep=0):
    cases=[];coords=[];times=[];pairid=[];sides=[];conditions=[]
    signs=[(1,None),(-1,None),(1,1),(1,-1),(-1,1),(-1,-1)]
    for i,pair in enumerate(layouts['axes']):
        kind=str(layouts['types'][i]).split('_negative')[0].split('_main')[0]
        raxis=axes(48,kind,1,c.CONFIG['role_seeds']['low_k'],i,f'low_random_coordinates_{rep}')[0]
        for side in range(2):
            a=raxis if arm=='I-R' else pair[side]
            coord=coordinates(a[None],arm)[0]
            for j,(v1,v2) in enumerate(signs):
                x=np.full((48,48),2,dtype=np.int8);x[23,23]=(v1+1)//2
                if v2 is not None:x[23,25]=(v2+1)//2
                cases.append(x);coords.append(coord);times.append(1-(1 if v2 is None else 2)/2304);pairid.append(i);sides.append(side);conditions.append(j)
    n=len(cases)
    return dict(noisy=np.array(cases),input_coordinates=np.array(coords),t=np.array(times,dtype=np.float32),
                queries=np.full((n,1),23*48+22,dtype=np.int64),labels=np.zeros((n,1),dtype=np.int8),
                parent=np.array(pairid),chain=np.array(sides),physical_axes=np.array([layouts['axes'][i,s] for i,s in zip(pairid,sides)]),
                width=np.array(48),kind=np.array('low_k_constructed'),pair=np.array(pairid),side=np.array(sides),condition=np.array(conditions))

def low_reference(parent,layouts,check=lambda:None):
    n=len(parent.spins);out=np.empty((n,80,2,8),dtype=np.int64)
    # Common translations within a parent and across both sides. Negative pairs
    # therefore have exactly equal empirical joint tables, not merely equal means.
    origins=np.stack([c.rng(c.CONFIG['role_seeds']['low_k'],i,'oracle_translation').integers(1024,size=(256,2)) for i in range(n)])
    for pair in range(80):
        check()
        for side in range(2):
            a=layouts['axes'][pair,side];positions=[(a[0,23],a[1,j]) for j in [22,23,25]]
            for start in range(0,n,32):
                rows=np.arange(start,min(start+32,n));o=origins[rows];code=np.zeros(o.shape[:2],dtype=np.int64)
                for (dx,dy),bit in zip(positions,[4,2,1]):
                    code+=bit*(parent.spins[rows[:,None],(o[:,:,0]+dx)%1024,(o[:,:,1]+dy)%1024]>0)
                out[rows,pair,side]=np.array([np.bincount(v,minlength=8) for v in code])
    assert np.all(out.sum(-1)==256)
    for i,label in enumerate(layouts['types']):
        if str(label).endswith('negative'):assert np.array_equal(out[:,i,0],out[:,i,1])
    return dict(counts=out,chain=np.asarray(parent.chain_ids),parent=np.arange(n),origin=origins)
