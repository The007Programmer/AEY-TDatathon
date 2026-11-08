import pandas as pd, numpy as np
P="kit/public/"
q=pd.read_csv(P+"validation/queries.csv"); t=pd.read_csv(P+"validation/targets.csv"); c=pd.read_csv(P+"validation/conditions.csv")
d=q.merge(t.drop(columns="episode_id"),on="query_id").merge(c,on="query_id")
pc=["p_0","p_1","p_2","p_5","p_20"]
print(d.groupby(["machine_type"])[pc].agg(["mean","std"]).round(3))
print(d.groupby(["signal"])[pc].mean().round(3))
print(d.groupby(["condition"])[pc].mean().round(3))
# conditional on winning: payout-size dist
w=d[pc[1:]].div(1-d.p_0,axis=0)
d2=pd.concat([d[["machine_type","signal"]],w],axis=1)
print("win-size dist by type"); print(d2.groupby("machine_type")[pc[1:]].agg(["mean","std"]).round(3))
print(d[pc].head(16).round(3).assign(t=d.machine_type.head(16),s=d.signal.head(16)))
h=pd.read_csv(P+"train/history.csv")
print(h.groupby("machine_id").size().describe())
print(h.payout.value_counts(normalize=True))
