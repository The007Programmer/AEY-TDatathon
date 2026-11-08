import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys,json; sys.path.insert(0,str(ROOT/'entry')); sys.path.insert(0,str(ROOT/'work'))
import hemodel, predict; from common import report
import pandas as pd
if __name__=="__main__":
    q,H=hemodel.load(str(ROOT/'kit/public/validation'))
    w,s=predict.read_config(str(ROOT/'entry/config.json'))
    s2=dict(s,mode="dirichlet",dir_prior={"steady":[.6,.31,.08,.01],"balanced":[.37,.44,.16,.03],"jackpot":[.24,.28,.36,.12]})
    for nm,ss in (("dirichlet fallback",s2),("family (expect 0.0585)",s)):
        out=hemodel.predict_all(q,H,w,ss,workers=12)
        report(pd.DataFrame([[a,*b] for a,b in out],columns=["query_id","p_0","p_1","p_2","p_5","p_20"]),nm)
