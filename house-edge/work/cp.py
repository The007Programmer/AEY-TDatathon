import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,str(ROOT/'work'))
from common import *
from scipy.special import betaln, gammaln
PAIRS=list(itertools.combinations(range(1,9),2))

def machine_arrays(s, cutoff, pr):
    """ticks t<cutoff where y and both lags known. returns ticks, cell, y"""
    T=[];C=[];Y=[]
    for t in range(8,cutoff):
        v=s.get(t)
        if v is None: continue
        a=s.get(t-pr[0]); b=s.get(t-pr[1])
        if a is None or b is None: continue
        T.append(t); C.append(2*(a>0)+(b>0)); Y.append(v>0)
    return np.array(T,int),np.array(C,int),np.array(Y,int)

def cum_counts(T,C,Y,grid):
    # counts[g,cell,y] for ticks < grid[g]
    out=np.zeros((len(grid),4,2))
    for g,tau in enumerate(grid):
        m=T<tau
        np.add.at(out[g],(C[m],Y[m]),1)
    return out

def bb(k1,k0,a,b):  # beta-binomial marginal log-lik (sequence)
    return betaln(a+k1,b+k0)-betaln(a,b)

class CP:
    def __init__(s, a0=0.5, S=6.0, lam=1.0, step=4, pchange=0.5, tmin=40):
        s.a0=a0; s.S=S; s.lam=lam; s.step=step; s.pchange=pchange; s.tmin=tmin
    def machine_ll(s, cnt_pre, cnt_all):
        """cnt_pre: (G,4,2) counts before tau, cnt_all: (4,2). returns ll[G] and post-pred win prob per cell [G,4]"""
        a=s.a0
        k1p=cnt_pre[...,1]; k0p=cnt_pre[...,0]
        llpre=bb(k1p,k0p,a,a).sum(-1)
        mpre=(k1p+a)/(k1p+k0p+2*a)
        # post prior centered on flipped pre mean
        A=s.S*(1-mpre)+0.3; B=s.S*mpre+0.3
        k1q=cnt_all[None,:,1]-k1p; k0q=cnt_all[None,:,0]-k0p
        llpost=bb(k1q,k0q,A,B).sum(-1)
        ppost=(k1q+A)/(k1q+k0q+A+B)
        return llpre+llpost, ppost
    def predict_episode(s, rows, H):
        cutoff=rows[0].cutoff
        grid=np.arange(s.tmin, cutoff-1, s.step)
        G=len(grid)
        per=[]
        for r in rows:
            hist=H[r.machine_id]
            best=None
            for pr in PAIRS:
                T,C,Y=machine_arrays(hist,cutoff,pr)
                if len(T)<10: continue
                cnt=cum_counts(T,C,Y,np.append(grid,cutoff+1))
                ll,pp=s.machine_ll(cnt[:-1],cnt[-1])
                llnone=bb(cnt[-1][:,1],cnt[-1][:,0],s.a0,s.a0).sum()
                pnone=(cnt[-1][:,1]+s.a0)/(cnt[-1].sum(-1)+2*s.a0)
                score=np.logaddexp(llnone, np.log(1e-300+0)+0) # placeholder
                tot=np.logaddexp(llnone+np.log(1-s.pchange), np.log(s.pchange/G)+np.logaddexp.reduce(ll))
                if best is None or tot>best[0]:
                    a=hist.get(cutoff-pr[0]); b=hist.get(cutoff-pr[1])
                    cell=None if a is None or b is None else 2*(a>0)+(b>0)
                    best=(tot,pr,ll,llnone,pp,pnone,cell)
            per.append(best)
        # episode posterior over tau: sum of machine ll (weighted by lam) + own
        ep_ll=sum(b[2] for b in per); ep_none=sum(b[3] for b in per)
        out={}
        for r,b in zip(rows,per):
            tot,pr,ll,llnone,pp,pnone,cell=b
            lt=ll+s.lam*(ep_ll-ll); ln=llnone+s.lam*(ep_none-llnone)
            lt=lt+np.log(s.pchange/G); ln=ln+np.log(1-s.pchange)
            m=max(lt.max(),ln); w=np.exp(lt-m); wn=np.exp(ln-m); Z=w.sum()+wn
            if cell is None:
                pw=None
            else:
                pw=(w@pp[:,cell]+wn*pnone[cell])/Z
            out[r.query_id]=(pw,pr,cell)
        return out

def size_dist(hist,cutoff,mtype,hl=60):
    pri={"steady":[.6,.31,.08,.01],"balanced":[.37,.44,.16,.03],"jackpot":[.24,.28,.36,.12]}[mtype]
    c=np.array(pri)*8.0
    for t,v in hist.items():
        if t<cutoff and v>0: c[PAY.index(v)-1]+=2**(-(cutoff-1-t)/hl)
    return c/c.sum()

def run(sp="validation", **kw):
    q,H=load(sp)
    kw=dict(kw); fam=kw.pop("fam",True); m=CP(**kw); kw["fam"]=fam
    res=[]
    for ep,g in q.groupby("episode_id"):
        rows=list(g.itertuples())
        o=m.predict_episode(rows,H)
        for r in rows:
            pw,pr,cell=o[r.query_id]
            hist=H[r.machine_id]
            if pw is None:
                ys=[hist[t]>0 for t in hist if t<r.cutoff]; pw=np.mean(ys[-30:])
            sd=size_family(hist,r.cutoff,r.machine_type) if kw.get("fam",True) else size_dist(hist,r.cutoff,r.machine_type)
            p=np.concatenate([[1-pw],pw*sd])
            p=np.clip(p,1e-6,1); p/=p.sum()
            res.append([r.query_id,*p])
    return pd.DataFrame(res,columns=["query_id"]+PC)

# exact size family: log(s5/s2), log(s20/s5) fixed per type; theta=log(s2/s1) ~ N(mu,0.65)
FAM={"steady":(-0.705,-1.406,-3.892),"balanced":(0.210,-0.981,-1.792),"jackpot":(0.265,0.251,-1.099)}
TH=np.linspace(-3.5,3.5,281)
def size_family(hist,cutoff,mtype,sd=0.65,hl=None):
    mu,r52,r205=FAM[mtype]
    w2=1.0; w5=np.exp(r52); w20=w5*np.exp(r205)
    big=w2+w5+w20
    n1=nb=0.0
    for t,v in hist.items():
        if t<cutoff and v>0:
            wt=1.0 if hl is None else 2**(-(cutoff-1-t)/hl)
            if v==1: n1+=wt
            else: nb+=wt
    p1=1/(1+np.exp(TH)*big)
    lp=-0.5*((TH-mu)/sd)**2+n1*np.log(p1)+nb*np.log1p(-p1)
    w=np.exp(lp-lp.max()); w/=w.sum()
    P1=w@p1
    rest=(1-P1)*np.array([w2,w5,w20])/big
    return np.array([P1,*rest])

if __name__=="__main__":
    import time; t0=time.time()
    pr=run()
    print(time.time()-t0)
    report(pr,"cp")

