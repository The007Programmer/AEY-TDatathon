import sys; sys.path.insert(0,'work')
from cp import *
from collections import Counter
q,H=load("train")
cands=PAIRS+[(l,) for l in range(1,9)]
cnt=Counter(); gains=[]
def ll(k1,n): 
    p=(k1+.5)/(n+1); return k1*np.log(p)+(n-k1)*np.log(1-p)
for r in q.itertuples():
    s=H[r.machine_id]
    wt=[t for t,v in s.items() if v>0 and t>=8]
    y=np.array([s[t]==1 for t in wt])
    base=ll(y.sum(),len(y))
    best=(-1e9,None)
    for pr in cands:
        # binarize lags: win/loss  ; also try "was 1" 
        for mode in ("win",):
            keys=[]; ok=[]
            for t in wt:
                v=[s.get(t-l) for l in pr]
                if None in v: ok.append(False); keys.append(None); continue
                ok.append(True); keys.append(tuple(int(x>0) for x in v))
            ok=np.array(ok,dtype=bool)
            if ok.sum()<5: continue
            L=0;k=0
            for key in set(kk for kk in keys if kk is not None):
                m=np.array([kk==key for kk in keys])
                L+=ll(y[m].sum(),m.sum()); k+=1
            L0=ll(y[ok].sum(),ok.sum())
            g=(L-L0)-0.5*(k-1)*np.log(max(ok.sum(),2))
            if g>best[0]: best=(g,pr)
    cnt[best[1] if best[0]>2 else None]+=1; gains.append(best[0])
print(cnt.most_common(15)); print(np.percentile(gains,[10,50,90]))
