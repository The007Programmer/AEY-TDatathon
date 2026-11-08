import sys; sys.path.insert(0,'work')
from cp import *
from collections import Counter
for sp in ["train","validation"]:
    q,H=load(sp); c=pd.read_csv(P+sp+"/conditions.csv").set_index("query_id")
    rows=[]
    for r in q.itertuples():
        if c.loc[r.query_id].condition not in ("stable",): continue
        s=H[r.machine_id]; best=None
        for pr in PAIRS:
            T,C,Y=machine_arrays(s,r.cutoff,pr)
            k=np.zeros((4,2)); np.add.at(k,(C,Y),1)
            L=bb(k[:,1],k[:,0],.5,.5).sum()
            if best is None or L>best[0]:
                rate=(k[:,1]+.5)/(k.sum(1)+1)
                best=(L,pr,rate)
        L,pr,rate=best
        xn=(rate[0]+rate[3])-(rate[1]+rate[2])  # XNOR effect
        rows.append(dict(type=r.machine_type,sig=r.signal,pair=pr,xn=xn,r00=rate[0],r01=rate[1],r10=rate[2],r11=rate[3],mid=r.machine_id[-4:]))
    D=pd.DataFrame(rows)
    print(sp); print(D.pair.value_counts().head(8))
    print(pd.crosstab([D.type,D.sig],np.sign(D.xn)))
    print(D.groupby(["type","sig"])[["r00","r01","r10","r11"]].mean().round(2))
    print(pd.crosstab(D.mid,D.pair).iloc[:8])
