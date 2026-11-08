import sys; sys.path.insert(0,'work')
from cp import *
q,H=load("train")
cat=lambda v: 0 if v==0 else (1 if v==1 else 2)
import itertools
# for each machine, best single lag (cat3) by LL gain; print share table
def ll(k1,n):
    if n==0: return 0
    p=(k1+.5)/(n+1); return k1*np.log(p)+(n-k1)*np.log(1-p)
from collections import Counter
C=Counter(); shown=0
for r in q.itertuples():
    s=H[r.machine_id]
    best=(0,None,None)
    for l in range(1,9):
        wt=[t for t,v in s.items() if v>0 and s.get(t-l) is not None]
        y=np.array([s[t]==1 for t in wt]); k=np.array([cat(s[t-l]) for t in wt])
        g=sum(ll(y[k==c].sum(),(k==c).sum()) for c in set(k))-ll(y.sum(),len(y))
        if g>best[0]: best=(g,l,[(round(y[k==c].mean(),2),(k==c).sum()) for c in range(3)])
    if best[0]>4:
        C[(r.machine_type,best[1])]+=1
        if shown<25: print(r.machine_type,round(best[0],1),best[1],best[2]); shown+=1
print(C)
