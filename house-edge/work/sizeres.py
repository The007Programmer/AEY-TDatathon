import sys; sys.path.insert(0,'work')
from common import *
import numpy as np, pandas as pd
pr=pd.read_csv("runs/entry_val.csv").set_index("query_id")
q,H=load("validation"); t=pd.read_csv(P+"validation/targets.csv").set_index("query_id"); c=pd.read_csv(P+"validation/conditions.csv").set_index("query_id")
rows=[]
for r in q.itertuples():
    T=t.loc[r.query_id]; Pp=pr.loc[r.query_id]; s=H[r.machine_id]
    tr=np.log(T.p_2/T.p_1); prr=np.log(Pp.p_2/Pp.p_1)
    d=dict(cond=c.loc[r.query_id].condition,type=r.machine_type,tr=tr,pr=prr,tw=1-T.p_0,sig=r.signal)
    for l in range(1,9):
        v=s.get(r.cutoff-l); d[f"b{l}"]=np.nan if v is None else int(v>=2); d[f"o{l}"]=np.nan if v is None else int(v==1); d[f"w{l}"]=np.nan if v is None else int(v>0)
    rows.append(d)
D=pd.DataFrame(rows); D["res"]=D.tr-D.pr
print("corr(true r21, our r21) by cond:",D.groupby("cond").apply(lambda x: np.corrcoef(x.tr,x.pr)[0,1]).round(3).to_dict())
print("resid sd by cond:",D.groupby("cond").res.std().round(3).to_dict())
for ty,x in D.groupby("type"):
    print(ty,"corr res~tw",round(np.corrcoef(x.res,x.tw)[0,1],3),"res~sig",round(np.corrcoef(x.res,x.sig)[0,1],3),
      {k:round(x[[k,"res"]].dropna().corr().iloc[0,1],2) for k in [f"{p}{l}" for p in "bow" for l in range(1,9)]})
