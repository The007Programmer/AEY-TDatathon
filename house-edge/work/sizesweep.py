import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys, json; sys.path.insert(0,str(ROOT/'entry')); sys.path.insert(0,str(ROOT/'work'))
import hemodel, predict
from common import report
import pandas as pd, numpy as np
from multiprocessing import Pool
from collections import defaultdict
q,H=hemodel.load(str(ROOT/'kit/public/validation'))
win,size=predict.read_config(str(ROOT/'entry/config.json'))
eps=defaultdict(list)
for r in q: eps[r["episode_id"]].append(r)
def job(a):
    ep,wo,so=a; w=dict(win,**wo); s=dict(size,**so)
    rows=eps[ep]; return hemodel.predict_all(rows,H,w,s)
if __name__=="__main__":
    confs=json.loads(sys.argv[1])
    with Pool(12) as p:
        for c in confs:
            so=c.get("size",{})
            if "lag_prior" in so: so["lag_prior"]={int(k):v for k,v in so["lag_prior"].items()}
            out=p.map(job,[(e,c.get("win",{}),so) for e in eps])
            df=pd.DataFrame([[qid,*pp] for o in out for qid,pp in o],columns=["query_id","p_0","p_1","p_2","p_5","p_20"])
            report(df,json.dumps(c))
