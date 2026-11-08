import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import pandas as pd, numpy as np
P=str(ROOT/'kit/public')+'/'
PC=["p_0","p_1","p_2","p_5","p_20"]; PAY=[0,1,2,5,20]
def load(sp):
    q=pd.read_csv(P+sp+"/queries.csv"); h=pd.read_csv(P+sp+"/history.csv")
    H={}
    for (e,m),g in h.groupby(["episode_id","machine_id"]):
        H[m]=dict(zip(g.tick.values.tolist(),g.payout.values.tolist()))
    return q,H
def score(pred, sp="validation"):
    """pred: DataFrame query_id + PC. returns R, per-cond ratios"""
    t=pd.read_csv(P+sp+"/targets.csv"); c=pd.read_csv(P+sp+"/conditions.csv"); ref=pd.read_csv(P+sp+"/reference.csv")
    d=t.merge(c,on="query_id")
    e=lambda df: ((df.set_index("query_id").loc[d.query_id,PC].values-d[PC].values)**2).sum(1)
    d["e"]=e(pred); d["b"]=e(ref)
    ep=d.groupby(["condition","episode_id"])[["e","b"]].mean().groupby("condition").mean()
    ep["ratio"]=ep.e/ep.b
    R=ep.ratio.mean()
    return R, ep, d
def report(pred,name=""):
    R,ep,d=score(pred)
    print(f"{name}: R={R:.4f} pts60={60/(1+R):.2f} | "+" ".join(f"{k}:{v:.3f}" for k,v in ep.ratio.items()))
    return R,d
