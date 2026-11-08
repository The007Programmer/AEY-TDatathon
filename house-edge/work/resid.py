import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from common import *
import numpy as np, pandas as pd
pr=pd.read_csv(str(ROOT/'runs/entry_val.csv')).set_index("query_id")
q,H=load("validation"); t=pd.read_csv(P+"validation/targets.csv").set_index("query_id"); c=pd.read_csv(P+"validation/conditions.csv").set_index("query_id")
rows=[]
for r in q.itertuples():
    s=H[r.machine_id]; tw=1-t.loc[r.query_id].p_0; pw=1-pr.loc[r.query_id].p_0
    nb=[m for m in q[q.episode_id==r.episode_id].machine_id if m!=r.machine_id]
    d=dict(cond=c.loc[r.query_id].condition,type=r.machine_type,sig=r.signal,res=tw-pw,lres=np.log(tw/(1-tw))-np.log(pw/(1-pw)),tw=tw,pw=pw)
    for l in range(1,9):
        v=s.get(r.cutoff-l); d[f"o{l}"]=np.nan if v is None else int(v>0)
        vals=[H[m].get(r.cutoff-l) for m in nb]; vals=[int(x>0) for x in vals if x is not None]
        d[f"n{l}"]=np.mean(vals) if vals else np.nan
    rows.append(d)
D=pd.DataFrame(rows)
for cond,x in D.groupby("cond"):
    print(cond, "mean|res|",x.res.abs().mean().round(3), "corr:", {k:round(x[[k,"res"]].corr().iloc[0,1],2) for k in [f"o{l}" for l in range(1,9)]+[f"n{l}" for l in range(1,9)]+["sig"]})
# calibration: bins of pw
D["bin"]=pd.cut(D.pw,[0,.05,.15,.3,.5,.7,.85,.95,1])
print(D.groupby(["bin"]).agg(n=("tw","size"),pw=("pw","mean"),tw=("tw","mean")).round(3))
print(D[D.cond=="stable"].assign(a=lambda x:x.res.abs()).sort_values("a").tail(10)[["type","tw","pw"]])
