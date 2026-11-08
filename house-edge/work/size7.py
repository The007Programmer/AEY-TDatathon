import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from common import *
import numpy as np, pandas as pd
q,H=load("train")
def ll(y):
    n=len(y); 
    if n==0: return 0
    k=y.sum(); p=(k+.5)/(n+1); return k*np.log(p)+(n-k)*np.log(1-p)
G={"big2":0,"split":0,"lossvs1":0}; n=0
for r in q.itertuples():
    s=H[r.machine_id]
    best=None
    for l in (1,2,4):
        wt=[t for t,v in s.items() if v>0 and (t-l) in s]
        y=np.array([s[t]==1 for t in wt]); u=np.array([s[t-l] for t in wt])
        L=ll(y[u>=2])+ll(y[u<2])
        if best is None or L>best[0]: best=(L,l,y,u)
    L,l,y,u=best
    G["big2"]+=0
    G["split"]+=ll(y[u==2])+ll(y[u>=5])+ll(y[u<2])-L
    G["lossvs1"]+=ll(y[u>=2])+ll(y[u==0])+ll(y[u==1])-L
    n+=1
print({k:round(v/n,3) for k,v in G.items()}, "chance ~0.5")
