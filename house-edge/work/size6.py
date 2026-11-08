import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from cp import *
from collections import Counter
q,H=load("train")
def ll(k1,n):
    p=(k1+.5)/(n+1); return k1*np.log(p)+(n-k1)*np.log(1-p)
cands=[()]+[(l,) for l in range(1,9)]+PAIRS
for feat in ["big","win"]:
    C=Counter()
    for r in q.itertuples():
        s=H[r.machine_id]
        wt=[t for t,v in s.items() if v>0 and t>=8 and all(s.get(t-l) is not None for l in range(1,9))]
        y=np.array([s[t]==1 for t in wt]); n=len(y)
        best=(-1e9,None)
        for pr in cands:
            f=(lambda v: v>=2) if feat=="big" else (lambda v: v>0)
            k=np.array([sum(int(f(s[t-l]))<<i for i,l in enumerate(pr)) for t in wt]) if pr else np.zeros(n,int)
            L=sum(ll(y[k==c].sum(),(k==c).sum()) for c in set(k))
            b=L-0.5*len(set(k))*np.log(n)
            if b>best[0]: best=(b,pr)
        C[best[1]]+=1
    print(feat, C.most_common(12))
