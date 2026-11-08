import sys; sys.path.insert(0,'work')
from cp import *
from collections import Counter
q,H=load("validation")
t=pd.read_csv(P+"validation/targets.csv").set_index("query_id")
c=pd.read_csv(P+"validation/conditions.csv").set_index("query_id")
# Within a machine, the "size mean" of targets: E[payout|win]
rows=[]
for r in q.itertuples():
    s=H[r.machine_id]; T=t.loc[r.query_id]
    w=1-T.p_0; ev=(T.p_1*1+T.p_2*2+T.p_5*5+T.p_20*20)/w
    lags={f"L{l}":(None if s.get(r.cutoff-l) is None else int(s[r.cutoff-l]>0)) for l in range(1,9)}
    lagsv={f"V{l}":s.get(r.cutoff-l) for l in range(1,9)}
    # machine empirical size mean
    wins=[v for k,v in s.items() if v>0]
    rows.append(dict(q=r.query_id,type=r.machine_type,sig=r.signal,cond=c.loc[r.query_id].condition,ev=ev,emp=np.mean(wins),
        s1=T.p_1/w,s2=T.p_2/w,s5=T.p_5/w,s20=T.p_20/w,**lags,**lagsv))
D=pd.DataFrame(rows)
for ty in ["steady","balanced","jackpot"]:
    x=D[D.type==ty]
    print(ty, "corr ev~emp", np.corrcoef(x.ev,x.emp)[0,1].round(3), "ev std",x.ev.std().round(3))
    for l in range(1,9):
        g=x.groupby(f"L{l}").ev.mean()
        print(f"  L{l}",g.round(3).to_dict(), end="")
    print()
    print(x.groupby("sig").ev.mean().round(3).to_dict())
# ratios s2/s1, s5/s2 etc. - is the size dist a 1-param family?
D["r21"]=np.log(D.s2/D.s1); D["r52"]=np.log(D.s5/D.s2); D["r205"]=np.log(D.s20/D.s5)
print(D.groupby("type")[["r21","r52","r205"]].agg(["mean","std"]).round(3))
print(D[D.type=="steady"][["r21","r52","r205"]].corr().round(2))
