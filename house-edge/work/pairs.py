import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import pandas as pd, numpy as np, itertools
P=str(ROOT/'kit/public')+'/'
sp="validation"
q=pd.read_csv(P+sp+"/queries.csv"); t=pd.read_csv(P+sp+"/targets.csv"); c=pd.read_csv(P+sp+"/conditions.csv")
d=q.merge(t.drop(columns="episode_id"),on="query_id").merge(c,on="query_id")
h=pd.read_csv(P+sp+"/history.csv")
H={k:dict(zip(g.tick,g.payout)) for k,g in h.groupby("machine_id")}
pairs=list(itertools.combinations(range(1,9),2))+[(i,) for i in range(1,9)]
rows=[]
for r in d.itertuples():
    s=H[r.machine_id]; cut=r.cutoff
    best=None
    res={}
    for pr in pairs:
        cnt={}
        for tk,v in s.items():
            ctx=tuple(s.get(tk-l) for l in pr)
            if None in ctx: continue
            key=tuple(int(x>0) for x in ctx)
            a=cnt.setdefault(key,[0,0]); a[int(v>0)]+=1
        ll=sum(a[1]*np.log((a[1]+.5)/(sum(a)+1))+a[0]*np.log((a[0]+.5)/(sum(a)+1)) for a in cnt.values())
        k=len(cnt)
        bic=ll-0.5*k*np.log(max(1,sum(sum(a) for a in cnt.values())))
        ctx=tuple(s.get(cut-l) for l in pr)
        pred=None
        if None not in ctx:
            a=cnt.get(tuple(int(x>0) for x in ctx),[0,0]); pred=(a[1]+.5)/(sum(a)+1)
        res[pr]=(bic,pred)
    ok=[(b,pr,p) for pr,(b,p) in res.items() if p is not None]
    ok.sort(reverse=True)
    b,pr,p=ok[0] if ok else (None,None,np.nan)
    rows.append((r.query_id,r.condition,r.machine_type,1-r.p_0,p,pr, ok[0][0]-ok[1][0] if len(ok)>1 else 0))
R=pd.DataFrame(rows,columns=["q","cond","type","truewin","pred","pair","margin"])
R["err"]=(R.truewin-R.pred)**2
print(R.groupby("cond").err.mean(), R.err.mean())
print(R.pair.value_counts().head(15))
print(R.groupby("type").err.mean())
print(R.sort_values("err",ascending=False).head(15))
