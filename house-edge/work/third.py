import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from common import *
import numpy as np, pandas as pd
from collections import Counter
sp="train"; q,H=load(sp); c=pd.read_csv(P+sp+"/conditions.csv").set_index("query_id")
PAIRS=[(1,3),(2,5),(4,7)]
def ll(y):
    n=len(y); k=y.sum(); p=(k+.5)/(n+1); return k*np.log(p)+(n-k)*np.log(1-p)
gain=Counter(); tot={}
for r in q.itertuples():
    if c.loc[r.query_id].condition!="stable": continue
    s=H[r.machine_id]; W={t:int(v>0) for t,v in s.items()}
    best=None
    for pr in PAIRS:
        ts=[t for t in range(8,r.cutoff) if all(t-l in W for l in range(0,9))]
        y=np.array([W[t] for t in ts]); x=np.array([W[t-pr[0]]==W[t-pr[1]] for t in ts])
        L=ll(y[x])+ll(y[~x])
        if best is None or L>best[0]: best=(L,pr,ts,y,x)
    L,pr,ts,y,x=best
    for feat in [f"w{l}" for l in range(1,9)]+[f"b{l}" for l in range(1,9)]+[f"one{l}" for l in range(1,9)]:
        l=int(feat[-1])
        if feat[0]=="w": z=np.array([W[t-l] for t in ts])
        elif feat[0]=="b": z=np.array([s[t-l]>=2 for t in ts])
        else: z=np.array([s[t-l]==1 for t in ts])
        z=z.astype(bool)
        L2=sum(ll(y[(x==a)&(z==b)]) for a in (0,1) for b in (0,1) if ((x==a)&(z==b)).sum()>0)
        tot[feat]=tot.get(feat,0)+(L2-L)
n=sum(1 for r in q.itertuples() if c.loc[r.query_id].condition=="stable")
print("machines",n,"chance gain per machine ~1.0")
print({k:round(v/n,2) for k,v in sorted(tot.items(),key=lambda kv:-kv[1])})
