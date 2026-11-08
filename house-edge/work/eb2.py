import sys, math; sys.path.insert(0,'./entry')
import numpy as np, json, os
from hemodel import load, _M, _B
from multiprocessing import Pool
cfg=json.load(open("./entry/config.json")); fam=cfg["size"]["fam"]
D="./kit/public"
recs=[]
for sp in ("train","validation","test"):
    q,H=load(os.path.join(D,sp))
    for r in q:
        s=H[r["machine_id"]]; past=[v for t,v in s.items() if t<r["cutoff"]]; qb=(sum(v>=2 for v in past)+1)/(len(past)+2)
        Ns=[]
        for l in (1,2,4):
            n=np.zeros((3,2))
            for t,v in s.items():
                if t>=r["cutoff"] or v<=0: continue
                u=s.get(t-l); n[2 if u is None else int(u>=2),int(v==1)]+=1
            Ns.append(n)
        recs.append((r["machine_type"],qb,Ns))
def tot(args):
    sdm,bm,sdb=args; T=0
    for ty,qb,Ns in recs:
        mu0,r52,r205=fam[ty]; big=1+math.exp(r52)+math.exp(r52+r205)
        lp0=-0.5*((_M-mu0)/sdm)**2+(np.logaddexp(-0.5*((_B-bm)/sdb)**2,-0.5*((_B+bm)/sdb)**2) if bm>0 else -0.5*(_B/sdb)**2)
        lp0-=np.logaddexp.reduce(lp0.ravel())
        p0=1/(1+np.exp(_M)*big); p1=1/(1+np.exp(_M+_B)*big); pu=qb*p1+(1-qb)*p0
        ev=[np.logaddexp.reduce((lp0+n[0,1]*np.log(p0)+n[0,0]*np.log1p(-p0)+n[1,1]*np.log(p1)+n[1,0]*np.log1p(-p1)+n[2,1]*np.log(pu)+n[2,0]*np.log1p(-pu)).ravel()) for n in Ns]
        T+=np.logaddexp.reduce(ev)-math.log(3)
    return args,T
if __name__=="__main__":
    grid=[(0.6,bm,sdb) for bm in (1.1,1.2,1.3,1.4) for sdb in (0.08,0.15,0.22)]
    with Pool(14) as p:
        for a,T in sorted(p.map(tot,grid),key=lambda x:-x[1]): print(a,round(T,1))
