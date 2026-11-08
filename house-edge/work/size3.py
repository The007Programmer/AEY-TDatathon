import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from cp import *
q,H=load("validation")
t=pd.read_csv(P+"validation/targets.csv").set_index("query_id")
c=pd.read_csv(P+"validation/conditions.csv").set_index("query_id")
R=[]
for r in q.itertuples():
    s=H[r.machine_id]; T=t.loc[r.query_id]
    true=np.log(T.p_2/T.p_1)
    row=dict(type=r.machine_type,sig=r.signal,cond=c.loc[r.query_id].condition,true=true)
    for nm,lo in [("all",0),("last100",r.cutoff-100),("last50",r.cutoff-50),("last25",r.cutoff-25)]:
        v=[x for tk,x in s.items() if tk>=lo and x>0]
        n1=sum(x==1 for x in v); nb=len(v)-n1
        row[nm]=np.log((nb+.5)/(n1+.5)); row["n"+nm]=len(v)
    R.append(row)
D=pd.DataFrame(R)
for cond,x in D.groupby("cond"):
    print(cond, {k:np.corrcoef(x.true,x[k])[0,1].round(3) for k in ["all","last100","last50","last25"]})
print(D.groupby(["type","sig"]).true.agg(["mean","std"]).round(2))
