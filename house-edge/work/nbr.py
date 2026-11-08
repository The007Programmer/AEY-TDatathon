import sys; sys.path.insert(0,'work')
from common import *
import numpy as np, pandas as pd, itertools
sp=sys.argv[1] if len(sys.argv)>1 else "train"
q,H=load(sp); c=pd.read_csv(P+sp+"/conditions.csv").set_index("query_id")
PAIRS=[(1,3),(2,5),(4,7),(1,2)]
res=[]
for ep,g in q.groupby("episode_id"):
    cond=c.loc[g.query_id.iloc[0]].condition
    if cond!="stable": continue
    mids=list(g.machine_id); cut=g.cutoff.iloc[0]
    W={m:{t:int(v>0) for t,v in H[m].items()} for m in mids}
    for m in mids:
        s=W[m]
        best=None
        for pr in PAIRS:
            ts=[t for t in range(8,cut) if t in s and t-pr[0] in s and t-pr[1] in s]
            cell=np.array([s[t-pr[0]]==s[t-pr[1]] for t in ts]); y=np.array([s[t] for t in ts])
            k=[(y[cell==z].sum()+.5)/((cell==z).sum()+1) for z in (0,1)]
            ll=sum(y[cell==z].sum()*np.log(k[z])+((cell==z).sum()-y[cell==z].sum())*np.log(1-k[z]) for z in (0,1))
            if best is None or ll>best[0]: best=(ll,ts,y,np.where(cell,k[1],k[0]))
        _,ts,y,p=best
        r=y-p
        row={"m":m}
        for l in range(1,9):
            nbv=np.array([np.mean([W[o].get(t-l,np.nan) for o in mids if o!=m]) for t in ts])
            ok=~np.isnan(nbv)
            row[f"n{l}"]=np.corrcoef(r[ok],nbv[ok])[0,1]
            # specific neighbor max abs corr
            cs=[]
            for o in mids:
                if o==m: continue
                v=np.array([W[o].get(t-l,np.nan) for t in ts]); ok=~np.isnan(v)
                if ok.sum()>20 and v[ok].std()>0: cs.append(np.corrcoef(r[ok],v[ok])[0,1])
            row[f"max{l}"]=np.max(np.abs(cs))
        row["n"]=len(ts)
        res.append(row)
D=pd.DataFrame(res)
print("mean corr resid~nbr mean:",D[[f"n{l}" for l in range(1,9)]].mean().round(3).to_dict())
print("mean |corr| :",D[[f"n{l}" for l in range(1,9)]].abs().mean().round(3).to_dict(), "null~",round(np.sqrt(2/np.pi)/np.sqrt(D.n.mean()),3))
print("max single nbr |corr|:",D[[f"max{l}" for l in range(1,9)]].mean().round(3).to_dict())
