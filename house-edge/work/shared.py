import sys; sys.path.insert(0,'work')
from common import *
pr=pd.read_csv("submissions/main_v2/validation_predictions.csv").set_index("query_id")
t=pd.read_csv(P+"validation/targets.csv").set_index("query_id")
q,H=load("validation"); qi=q.set_index("query_id")
lg=lambda p: np.log(p/(1-p))
PAIRS=[(1,3),(2,5),(4,7)]
def fit_machine(s,cut,lo):
    # MAP pair & cell rates using ticks in [lo,cut)
    best=None
    for pr_ in PAIRS:
        ts=[x for x in range(max(lo,8),cut) if x in s and x-pr_[0] in s and x-pr_[1] in s]
        if len(ts)<10: continue
        cell=np.array([int((s[x-pr_[0]]>0)==(s[x-pr_[1]]>0)) for x in ts]); y=np.array([int(s[x]>0) for x in ts])
        k=np.array([(y[cell==z].sum()+.5)/((cell==z).sum()+1) for z in (0,1)])
        ll=(y*np.log(k[cell])+(1-y)*np.log(1-k[cell])).sum()
        if best is None or ll>best[0]: best=(ll,pr_,k)
    return best
rows=[]
for ep,g in q.groupby("episode_id"):
    cut=g.cutoff.iloc[0]
    R={}
    for r in g.itertuples():
        s=H[r.machine_id]; b=fit_machine(s,cut,cut-80)
        if b is None: continue
        _,pr_,k=b
        for x in range(cut-30,cut):
            if x in s and x-pr_[0] in s and x-pr_[1] in s:
                p=k[int((s[x-pr_[0]]>0)==(s[x-pr_[1]]>0))]; R.setdefault(x,[]).append((int(s[x]>0)-p)/max(p*(1-p),0.05))
    for r in g.itertuples():
        tr=lg(1-t.loc[r.query_id].p_0)-lg(1-pr.loc[r.query_id].p_0)
        f={}
        for W in (3,6,10,20,30):
            v=[z for x in range(cut-W,cut) for z in R.get(x,[])]
            f[f"f{W}"]=np.mean(v) if v else 0
        rows.append(dict(ep=ep,res=tr,type=r.machine_type,sig=r.signal,**f))
D=pd.DataFrame(rows); D["res"]-=D.groupby("type").res.transform("mean")
print({k:round(np.corrcoef(D.res,D[k])[0,1],3) for k in D.columns if k.startswith("f")})
for k in ["f6","f10","f20"]:
    b=np.polyfit(D[k],D.res,1); print(k,"slope",b.round(3))
# apply adjustment and score
F={}
for ep,g in q.groupby("episode_id"):
    pass
D["qid"]=[r.query_id for ep,g in q.groupby("episode_id") for r in g.itertuples()]
for W in ("f3","f6","f10"):
  for k in (0.2,0.35,0.5):
    P2=pr.copy()
    f=D.set_index("qid")[W]
    for qid in P2.index:
        p=P2.loc[qid,PC].values.astype(float); w=1-p[0]
        w2=1/(1+np.exp(-(lg(np.clip(w,1e-6,1-1e-6))+k*f[qid])))
        P2.loc[qid,PC]=np.r_[1-w2,p[1:]/w*w2]
    R,_,_=score(P2.reset_index()); print(W,k,round(R,4))
