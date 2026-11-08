import sys,json; sys.path.insert(0,'./work')
import cp2
from common import *
from multiprocessing import Pool
import numpy as np, pandas as pd
Q,HH=load("validation")
def ep_job(args):
    ep,kw,skw=args
    g=Q[Q.episode_id==ep]; rows=list(g.itertuples()); m=cp2.CP2(**kw); o=m.predict_episode(rows,HH); res=[]
    for r in rows:
        pw=o[r.query_id]; sd=cp2.size_lag(HH[r.machine_id],r.cutoff,r.machine_type,**skw)
        p=np.concatenate([[1-pw],pw*sd]); p=np.clip(p,1e-7,1); p/=p.sum(); res.append([r.query_id,*p])
    return res
def runp(pool,kw={},skw={}):
    out=pool.map(ep_job,[(e,kw,skw) for e in Q.episode_id.unique()])
    return pd.DataFrame([x for o in out for x in o],columns=["query_id"]+PC)
if __name__=="__main__":
    configs=json.loads(sys.argv[1])
    with Pool(10) as pool:
        for c in configs:
            pr=runp(pool,c.get("kw",{}),c.get("skw",{}))
            report(pr,json.dumps(c))
