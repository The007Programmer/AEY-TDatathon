import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from common import *
import numpy as np, itertools
from size_model import FAM
# collect per machine per lag counts n[l][big][is1]
data=[]
for sp in ["train","validation","test"]:
    q,H=load(sp)
    for r in q.itertuples():
        s=H[r.machine_id]; N=np.zeros((8,2,2))
        for t,v in s.items():
            if v<=0: continue
            for l in range(1,9):
                u=s.get(t-l)
                if u is None: continue
                N[l-1,int(u>=2),int(v==1)]+=1
        data.append((r.machine_type,N))
MU=np.linspace(-3,3,61); BE=np.linspace(-4,4,81); M_,B_=np.meshgrid(MU,BE,indexing="ij")
def total_ll(sd_mu, bprior, lagw, mus):
    tot=0
    for ty,N in data:
        mu0,r52,r205=FAM[ty]; mu0=mus[ty]
        big=1+np.exp(r52)+np.exp(r52+r205)
        lp=-0.5*((M_-mu0)/sd_mu)**2+np.log(bprior(B_)+1e-300); lp-=np.logaddexp.reduce(lp.ravel())
        p0=1/(1+np.exp(M_)*big); p1=1/(1+np.exp(M_+B_)*big)
        ev=[]
        for l in range(8):
            n=N[l]
            L=lp+n[0,1]*np.log(p0)+n[0,0]*np.log1p(-p0)+n[1,1]*np.log(p1)+n[1,0]*np.log1p(-p1)
            ev.append(np.logaddexp.reduce(L.ravel())+np.log(lagw[l]))
        tot+=np.logaddexp.reduce(ev)
    return tot
mus={k:v[0] for k,v in FAM.items()}
gauss=lambda sd: (lambda b: np.exp(-0.5*(b/sd)**2))
def lagw(main, other): 
    w=np.array([main if l in (1,2,4) else other for l in range(1,9)]); return w/w.sum()
res=[]
for sdm in [0.3,0.45,0.6]:
    for sdb in [0.8,1.2,1.8]:
        for oth in [0.02,0.15]:
            L=total_ll(sdm,gauss(sdb),lagw(1,oth),mus); res.append((L,sdm,sdb,oth)); print(round(L,1),sdm,sdb,oth,flush=True)
# two-point + zero mixture
for b0 in [1.0,1.5,2.0]:
    for pz in [0.0,0.3]:
        f=lambda b,b0=b0,pz=pz: pz*np.exp(-0.5*(b/0.15)**2)+(1-pz)*(np.exp(-0.5*((b-b0)/0.3)**2)+np.exp(-0.5*((b+b0)/0.3)**2))
        L=total_ll(0.45,f,lagw(1,0.02),mus); print("2pt",round(L,1),b0,pz,flush=True)
