"""Reproducible RG-2 method schematics; no trained-model output is synthesized.

Run from any directory: python integrated/scripts/method_figures.py
The toy field illustrates exact input operators, not an MC measurement.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'integrated'
COLORS = {'MC':'#263e41', 'F':'#657895', 'P':'#89568e', 'D':'#1877a3', 'M':'#b65322'}
SPIN = ListedColormap(['#f3ecda', '#193f41'])
MASK = '#a9b5c5'
plt.rcParams.update({'font.family':['Arial','DejaVu Sans','sans-serif'], 'font.size':15,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':1.3,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})
CAPTIONS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canvas(size):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    ax.text(.025,.975,'METHOD SCHEMATIC · not measured output',va='top',fontsize=12,color='#65706c')
    return fig,ax


def box(ax,x,y,w,h,text,color='#263e41',fill='#f7f8f6',fontsize=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=.012',
                              facecolor=fill,edgecolor=color,linewidth=1.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=color,linespacing=1.35)


def arrow(ax,start,end,color='#566760',style='-|>',rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=17,
                                connectionstyle=f'arc3,rad={rad}',linewidth=1.7,color=color))


def save(fig,name,caption):
    path=OUT/'figures'/name
    for ext in ('png','pdf'):
        fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    CAPTIONS[name]=caption


def draw_spin(fig,rect,spins,title,coords=None,numbered=False,mask=None,outline=None):
    ax=fig.add_axes(rect)
    z=np.asarray(spins)
    if mask is None:
        ax.imshow(z,cmap=SPIN,vmin=-1,vmax=1,interpolation='nearest',origin='upper')
    else:
        display=np.where(mask,2,(z+1)//2)
        ax.imshow(display,cmap=ListedColormap(['#f3ecda','#193f41',MASK]),vmin=0,vmax=2,interpolation='nearest',origin='upper')
    n=z.shape[0]
    ax.set_xticks(np.arange(-.5,n,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='both',length=0)
    if coords is None:
        ax.set_xticks([]);ax.set_yticks([])
    else:
        ax.set_xticks(range(n),coords);ax.set_yticks(range(n),coords)
        ax.tick_params(labelsize=12)
    if numbered:
        for i in range(n):
            for j in range(n):
                label='?' if mask is not None and mask[i,j] else ('+' if z[i,j]>0 else '−')
                ax.text(j,i,label,ha='center',va='center',fontsize=17,
                        color='#263e41' if mask is not None and mask[i,j] else ('white' if z[i,j]>0 else '#263e41'))
    if outline is not None:
        ax.add_patch(Rectangle(outline[:2],outline[2],outline[3],fill=False,edgecolor='#b65322',linewidth=2.5,zorder=5))
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#9ba7a0');spine.set_linewidth(.8)
    ax.set_title(title,fontsize=16,pad=10)
    return ax


def toy_data():
    counts=np.array([[5,9,4],[6,3,7],[2,8,5]])
    center=np.array([[-1,1,1],[1,-1,-1],[-1,1,-1]])
    raw=np.empty((9,9),dtype=int)
    for i in range(3):
        for j in range(3):
            block=-np.ones(9,dtype=int)
            if center[i,j]>0:block[4]=1
            remaining=int(counts[i,j]-(center[i,j]>0))
            # Fixed cyclic order makes this exactly reproducible without a seed.
            order=[0,2,6,8,1,3,5,7]
            for k in order[:remaining]:block[k]=1
            raw[3*i:3*i+3,3*j:3*j+3]=block.reshape(3,3)
    f=raw[3:6,3:6].copy();d=raw[1::3,1::3].copy()
    m=np.sign(raw.reshape(3,3,3,3).sum((1,3))).astype(int)
    assert np.array_equal(((raw+1)//2).reshape(3,3,3,3).sum((1,3)),counts)
    assert np.array_equal(d,center)
    assert m[0,0]==m[0,1]==1 and counts[0,0]==5 and counts[0,1]==9
    return raw, f, d, m, counts


def construction():
    fig,ax=canvas((13.5,7.4))
    ax.text(.025,.905,'a   One training cycle, shown at W = 32 (18 samples per batch)',fontsize=18,weight='bold')
    box(ax,.035,.69,.20,.15,'L1024 periodic\ntraining parent pool')
    box(ax,.31,.725,.24,.10,'Common: new 32 × 32 crop\nunit coordinates, level 0',fontsize=15)
    box(ax,.69,.725,.27,.10,'Same input across F / P / D / M\n→ one common AdamW update',fontsize=15)
    arrow(ax,(.24,.785),(.30,.785));arrow(ax,(.56,.775),(.68,.775))
    box(ax,.31,.49,.24,.10,'Auxiliary: new 96 × 96 crop\nshared across the four arms',fontsize=15)
    arrow(ax,(.14,.68),(.30,.54))
    ax.text(.59,.545,'Separate role-keyed draw from common stream;\nnot a paired fine/coarse example.',fontsize=14,va='center')
    cards=[('F','Central 32 × 32 fine spins','(i, j)','32 × 32 fine crop'),
           ('P','Identical spins to F','(3i + 1, 3j + 1)','32 × 32 fine crop'),
           ('D','One center per 3 × 3 block','(3i + 1, 3j + 1)','96 × 96 block layout'),
           ('M','One majority bit per block','(3i + 1, 3j + 1)','96 × 96 block layout')]
    for j,(arm,spins,coords,footprint) in enumerate(cards):
        x=.025+j*.246
        box(ax,x,.185,.225,.20,f'{arm}\n{spins}\ncoords: {coords}\n{footprint}',color=COLORS[arm],fontsize=14)
        arrow(ax,(.43,.475),(x+.1125,.398),color=COLORS[arm])
    ax.text(.5,.105,'All auxiliary inputs: 32 × 32 tokens, level 1 → one auxiliary AdamW update',ha='center',fontsize=17,weight='bold')
    ax.text(.5,.05,'Next cycle uses the next W in 16 → 24 → 32 → 48; each role and arm keeps its prescribed pairing.',ha='center',fontsize=13)
    save(fig,'F03a_four_arm_construction','方法示意（非实测图）。以W32为例，每周期先执行共同细流更新，再执行辅助流更新。共同与辅助使用不同role随机键重新选择父场和原点；同一role内四臂严格配对。辅助F/P仅取96×96父窗中央32×32细自旋，D抽中心，M投票；所有辅助输入仅含32×32二值token、同一level1角色。物理覆盖与编码坐标不可混用。')

    raw,f,d,m,counts=toy_data()
    fig,ax=canvas((13.5,7))
    ax.text(.025,.905,'b   Hand-checkable 9 × 9 → 3 × 3 example (W = 3)',fontsize=18,weight='bold')
    parentax=draw_spin(fig,[.035,.40,.30,.40],raw,'Auxiliary parent window',numbered=True,outline=(2.5,2.5,3,3))
    for p in (2.5,5.5):
        parentax.axhline(p,color='#a29d91',linewidth=2);parentax.axvline(p,color='#a29d91',linewidth=2)
    arrays=[f,f,d,m]
    for j,(arm,arr) in enumerate(zip(('F','P','D','M'),arrays)):
        x=.385+j*.15
        draw_spin(fig,[x,.485,.115,.24],arr,arm,coords=[0,1,2] if arm=='F' else [1,4,7],numbered=True)
        ax.text(x+.0575,.38,'fine spins' if arm in ('F','P') else ('center spins' if arm=='D' else 'majority bits'),ha='center',color=COLORS[arm],fontsize=13)
    ax.text(.185,.32,'Orange square: central fine crop\nGrid lines: nonoverlapping 3 × 3 blocks',ha='center',fontsize=13)
    ax.text(.685,.29,'F and P have identical values; the coordinate axes differ.\nD and M share coordinates but may have different values.',ha='center',fontsize=15)
    box(ax,.035,.085,.40,.14,'Block positive-spin counts\n'+str(counts.tolist()).replace('], [',']   [').replace('[[','[').replace(']]',']'),fontsize=13)
    box(ax,.48,.085,.48,.14,'5:4 and 9:0 both become the same + token in M.\nNo vote count or nine-spin block is supplied to the network.',color=COLORS['M'],fontsize=15)
    save(fig,'F03b_hand_example','可手算方法示例，数值为人为固定构造，不是MC或模型输出。9×9父窗中的橙框为F/P取出的中央3×3；P只改变编码轴刻度。D取九个块中心，M取九块多数；每块正自旋数记录在图下。左上块5:4与上中块9:0都映射成M的同一正token，说明投票强度被丢弃。图中展示D4增强前坐标；示例完整数组保存在method_example.json。')
    return raw,f,d,m,counts


def model():
    fig,ax=canvas((13.5,7.8))
    ax.text(.025,.905,'a   Coordinate-aware masked Transformer · 1,976,962 trainable parameters',fontsize=18,weight='bold')
    box(ax,.035,.68,.18,.115,'Noisy tokens\n[B, W, W]',fontsize=16)
    box(ax,.275,.68,.19,.115,'Embedding 4 → 128\nflatten N = W²',fontsize=16)
    box(ax,.525,.68,.17,.115,'Add context h\nto every token',fontsize=16)
    box(ax,.76,.68,.20,.115,'7 Transformer blocks\n[B, N, 128]',fontsize=16)
    for start,end in [((.22,.738),(.265,.738)),((.47,.738),(.515,.738)),((.70,.738),(.75,.738))]:arrow(ax,start,end)
    box(ax,.25,.43,.30,.135,'h = MLP(sinusoidal time)\n+ task embedding[level]\ntime: 128 → 512 → 128',fontsize=16)
    arrow(ax,(.55,.51),(.61,.67));arrow(ax,(.55,.49),(.76,.69),rad=.15)
    box(ax,.035,.43,.15,.135,'t: [B]\nlevel: 0 or 1',fontsize=16);arrow(ax,(.195,.497),(.24,.497))
    box(ax,.735,.43,.24,.135,'LayerNorm → SiLU\nLinear 128 → 2\nlogits [B, 2, W, W]',fontsize=16)
    arrow(ax,(.86,.67),(.86,.575))
    box(ax,.035,.105,.255,.18,'Coordinates: [B, W, W, 2]\nRoPE acts only on Q and K\nNo direct coordinate input to V',color=COLORS['P'],fontsize=15)
    box(ax,.345,.105,.63,.18,'Each block: AdaLN(h) → full bidirectional attention → residual\nthen AdaLN(h) → MLP(128 → 512 → 128) → residual\n4 heads × 32 dimensions; Q/K LayerNorm; 2D RoPE; dropout 0',fontsize=16)
    arrow(ax,(.30,.195),(.335,.195),color=COLORS['P'])
    ax.text(.5,.04,'MASK tokens remain attention queries, keys and values. Both task embeddings start at zero.',ha='center',fontsize=13)
    save(fig,'F04a_model','实际模型结构的方法示意。token含负、正、MASK、PAD四类，本轮无PAD；七个全双向稠密注意力block共享同一时间/任务context。坐标只直接旋转Q、K，不直接注入V或残差流；MASK位置也参与注意力。AdaLN的scale/shift层与最终分类输出线性层零初始化，两个任务嵌入也零初始化。输出是每位置两类logits，不是联合自旋分布表。')


def mask_training():
    z=np.array([[1,1,1,-1,-1,-1],[1,1,-1,-1,1,-1],[1,-1,-1,1,1,-1],[-1,-1,1,1,1,1],[-1,1,1,1,-1,-1],[-1,1,1,-1,-1,-1]])
    mask=np.array([[0,1,0,1,0,0],[1,0,0,0,1,0],[0,0,1,0,1,0],[1,0,1,0,0,0],[0,1,0,0,1,0],[1,0,0,1,0,1]],dtype=bool)
    fig,ax=canvas((13.5,8))
    ax.text(.025,.905,'b   Masking and alternating training updates',fontsize=18,weight='bold')
    draw_spin(fig,[.035,.51,.22,.30],z,'Clean labels y',numbered=True)
    draw_spin(fig,[.37,.51,.22,.30],z,'Noisy input:  ? = MASK',numbered=True,mask=mask)
    targetax=fig.add_axes([.70,.51,.22,.30]);targetax.imshow(mask,cmap=ListedColormap(['#f7f8f6',COLORS['M']]),vmin=0,vmax=1,interpolation='nearest')
    targetax.set_xticks([]);targetax.set_yticks([]);targetax.set_title('Loss mask: supervise here',fontsize=16,pad=10)
    for i,j in zip(*np.where(mask)):
        targetax.text(j,i,'+' if z[i,j]>0 else '−',ha='center',va='center',color='white',fontsize=17)
    arrow(ax,(.265,.675),(.35,.675));arrow(ax,(.60,.675),(.68,.675))
    ax.text(.5,.445,'t ~ 0.98 Uniform[0.002, 1) + 0.02 δ₁;  maskᵢ | t ~ Bernoulli(t)',ha='center',fontsize=17)
    ax.text(.5,.39,'First construct F/P/D/M clean spins; then global spin flip, coordinate D4, and masking.',ha='center',fontsize=14)
    box(ax,.04,.185,.25,.13,'Common batch, level 0\nbackprop → AdamW → EMA',fontsize=16)
    box(ax,.375,.185,.25,.13,'Auxiliary batch, level 1\nbackprop → AdamW → EMA',fontsize=16)
    box(ax,.71,.185,.25,.13,'Next W in 16, 24, 32, 48\nrepeat paired cycle',fontsize=16)
    arrow(ax,(.30,.25),(.365,.25));arrow(ax,(.635,.25),(.70,.25))
    ax.text(.5,.105,r'Loss $= (BW^2)^{-1}\sum_b t_b^{-1}\sum_i m_{bi}[-\log p_\theta(y_{bi}\mid\widetilde y_b,t_b,c_b,\ell)]$',ha='center',fontsize=19)
    ax.text(.5,.04,'W = 32: B = 18 and BW² = 18,432.  Clean labels and loss mask are not model inputs.',ha='center',fontsize=14)
    save(fig,'F04b_mask_training','遮盖与优化方法示意；顶部6×6数组仅用于展示规则，正式示例W32时batch为18。先构造clean目标，再做全局自旋翻转及坐标D4，最后独立Bernoulli遮盖；t是样本级随机时间，另有2%全MASK端点。右侧只在隐藏位置监督，损失除以t并按整个batch所有token归一。共同与辅助分别进行独立优化器更新，而不是两loss相加后一次更新；两任务共享主干与同一AdamW状态。')
    return z,mask


def sampling(z):
    # Illustrate only reveal membership, with an explicit fixed toy final field.
    u=np.array([[.98,.25,.71,.43,.09,.61],[.55,.84,.34,.05,.95,.47],[.18,.69,.39,.77,.53,.13],
                [.89,.29,.63,.41,.01,.79],[.57,.93,.21,.49,.73,.33],[.07,.65,.45,.85,.15,.59]])
    fig,ax=canvas((13.5,7.9))
    ax.text(.025,.905,'c   Fine-field generation: the same procedure for F / P / D / M',fontsize=18,weight='bold')
    draw_spin(fig,[.035,.51,.22,.30],z,'Start: all MASK',numbered=True,mask=np.ones_like(z,dtype=bool))
    draw_spin(fig,[.37,.51,.22,.30],z,'Intermediate reveal state',numbered=True,mask=u<=.5)
    draw_spin(fig,[.70,.51,.22,.30],z,'End: every position revealed',numbered=True)
    arrow(ax,(.265,.675),(.35,.675));arrow(ax,(.60,.675),(.68,.675))
    ax.text(.5,.43,'Illustrative states only; the displayed ± values are not predictions from a trained model.',ha='center',fontsize=13,color='#65706c')
    box(ax,.035,.17,.285,.16,'Fixed inputs: W96, unit coordinates\nlevel 0, final 12k EMA\nS = 64 or 256; 32 images / setting',fontsize=15)
    box(ax,.36,.17,.285,.16,'Per position: independent Uᵢ and Vᵢ\ntⱼ = cos²(πj / 2S)\nreveal if tⱼ₊₁ < Uᵢ ≤ tⱼ',fontsize=15)
    box(ax,.685,.17,.285,.16,'Predict with clock max(tⱼ, 0.01)\nnew spin = 2 · 1[Vᵢ < pᵢ] − 1\npreviously revealed spins stay fixed',fontsize=15)
    ax.text(.5,.085,'The same U/V fields pair arms and S values; the 64-step grid is nested in the 256-step grid.',ha='center',fontsize=14)
    ax.text(.5,.035,'Fresh M does not generate or condition on a coarse field first.  Coarse statistics are computed afterward.',ha='center',fontsize=14,weight='bold')
    save(fig,'F04c_generation','从全MASK生成的规则示意，顶部6×6状态与符号仅为说明揭示过程的人为示例，不是训练模型生成结果。正式Fresh评价为W96、level0、单位坐标、64/256步、每设置32张，使用12k最终EMA。每位置预置独立U/V，U决定日程区间，V与当前条件概率比较决定自旋；组内并行揭示且不回改已揭示位。模型时钟下限0.01不截断揭示日程至0的进程。跨臂和步数复用随机身份，但不会保证最终图相同。')
    return u


def main():
    for name in ('figures','data','provenance'):(OUT/name).mkdir(parents=True,exist_ok=True)
    raw,f,d,m,counts=construction();model();z,mask=mask_training();u=sampling(z)
    record=dict(kind='deterministic_method_schematic_not_experimental_data',source_commit='5516370b3ae6e98160d9468668acdcbf851dc902',
                toy_9x9=raw.tolist(),F=f.tolist(),P=f.tolist(),D=d.tolist(),M=m.tolist(),block_positive_counts=counts.tolist(),
                coordinate_axes={'F':[0,1,2],'P':[1,4,7],'D':[1,4,7],'M':[1,4,7]},
                toy_mask_labels=z.tolist(),toy_loss_mask=mask.tolist(),toy_reveal_U=u.tolist(),
                assertions=dict(F_equals_P=True,centers_verified=True,majorities_verified=True,five_and_nine_votes_share_positive_token=True))
    (OUT/'data/method_example.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    sources=['project/scripts/rg2_v12/phase_b.py','project/scripts/rg2_v12/phase_a.py',
             'project/scripts/rg2_v12/pilot_net.py','project/ism_diffusion/scale_model.py','project/ism_diffusion/model.py']
    provenance=dict(source_commit=record['source_commit'],script='integrated/scripts/method_figures.py',script_sha256=digest(Path(__file__)),
                    source_sha256={p:digest(ROOT/p) for p in sources},toy_data='integrated/data/method_example.json',
                    toy_data_sha256=digest(OUT/'data/method_example.json'),
                    figures={name:dict(caption_zh=caption,kind='method_schematic',
                             png='integrated/figures/'+name+'.png',pdf='integrated/figures/'+name+'.pdf',
                             png_sha256=digest(OUT/'figures'/f'{name}.png'),pdf_sha256=digest(OUT/'figures'/f'{name}.pdf'))
                             for name,caption in CAPTIONS.items()})
    (OUT/'provenance/method_figure_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='complete',figures=list(CAPTIONS),all_outputs='PNG + PDF',toy_data_checks='passed')))


if __name__=='__main__':main()
