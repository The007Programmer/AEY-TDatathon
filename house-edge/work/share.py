import sys; sys.path.insert(0,'work')
from common import *
import numpy as np, pandas as pd
from collections import Counter
sp=sys.argv[1]; q,H=load(sp); c=pd.read_csv(P+sp+"/conditions.csv").set_index("query_id")
PAIRS=[(1,3),(2,5),(4,7),(1,2)]
info=[]
for r in q.itertuples():
    if c.loc[r.query_id].condition!="stable": continue
    s={t:int(v>0) for t,v in H[r.machine_id].items()}
    best=None
    for pr in PAIRS:
        ts=[t for t in range(8,r.cutoff) if t in s and t-pr[0] in s and t-pr[1] in s]
        cell=np.array([s[t-pr[0]]==s[t-pr[1]] for t in ts]); y=np.array([s[t] for t in ts])
        k=[(y[cell==z].sum()+.5)/((cell==z).sum()+1) for z in (0,1)]
        ll=sum(y[cell==z].sum()*np.log(k[z])+((cell==z).sum()-y[cell==z].sum())*np.log(1-k[z]) for z in (0,1))
        if best is None or ll>best[0]: best=(ll,pr,k)
    info.append(dict(ep=r.episode_id,idx=int(r.machine_id[-3:]),type=r.machine_type,sig=r.signal,pair=best[1],k0=best[2][0],k1=best[2][1]))
D=pd.DataFrame(info); D["dir"]=np.sign(D.k1-D.k0); D["pair"]=D.pair.astype(str)
print(pd.crosstab(D.idx,D.pair)); print(pd.crosstab(D.idx,D.dir)); print(pd.crosstab(D.type,D.pair))
print(pd.crosstab([D.type,D.sig],D.dir))
same=[]; 
for ep,g in D.groupby("ep"):
    p=list(g.pair); same+= [p[i]==p[j] for i in range(len(p)) for j in range(i+1,len(p))]
print("same-pair frac within episode",np.mean(same))
print(D.groupby(["type","dir"])[["k0","k1"]].mean().round(3))
