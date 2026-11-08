import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'./work')
from common import *
from scipy.special import betaln
from size_model import size_lag
PAIRS=list(itertools.combinations(range(1,9),2))
LEVELS={'steady':(0.94,0.15),'balanced':(0.86,0.05),'jackpot':(0.50,0.012)}
MAINP={(1,3):1.0,(2,5):1.0,(4,7):1.0,(1,2):0.1}
def bb(k1,k0,a,b): return betaln(a+k1,b+k0)-betaln(a,b)

def arrays(s,cutoff,pr):
    T=[];A=[];B=[];Y=[]
    for t in range(8,cutoff):
        v=s.get(t)
        if v is None: continue
        a=s.get(t-pr[0]); b=s.get(t-pr[1])
        if a is None or b is None: continue
        T.append(t);A.append(a>0);B.append(b>0);Y.append(v>0)
    return np.array(T,int),np.array(A,int),np.array(B,int),np.array(Y,int)

class CP2:
    def __init__(s,a0=0.5,S=6.0,lam=1.0,step=4,pchange=0.5,tmin=40,other=0.005,p4=0.3,xnor_only=False,lsh=0.0,tps=0.0,levels=None,mainp=None,hl=0):
        s.__dict__.update(locals()); del s.__dict__["s"]
    def evid(s,cnt_pre,cnt_all):
        a=s.a0
        k1p=cnt_pre[...,1];k0p=cnt_pre[...,0]
        llpre=bb(k1p,k0p,a,a).sum(-1)
        mpre=(k1p+a)/(k1p+k0p+2*a)
        A=s.S*(1-mpre)+0.3;B=s.S*mpre+0.3
        k1q=cnt_all[None,:,1]-k1p;k0q=cnt_all[None,:,0]-k0p
        return llpre+bb(k1q,k0q,A,B).sum(-1),(k1q+A)/(k1q+k0q+A+B)
    def evid_x(s,cnt_pre,cnt_all,mtype):
        """2-cell xnor model with type-level mixture prior over direction. cell0=diff, cell1=same"""
        hi,lo=(s.levels or LEVELS)[mtype]; n=s.tps
        k1p=cnt_pre[...,1];k0p=cnt_pre[...,0]
        comps=[]
        for d in (0,1):  # d=1: same->hi
            m=np.array([lo,hi]) if d==1 else np.array([hi,lo])
            A0=n*m; B0=n*(1-m)
            ll=bb(k1p,k0p,A0,B0).sum(-1)
            comps.append((ll,(k1p+A0)/(k1p+k0p+A0+B0)))
        L=np.logaddexp(comps[0][0],comps[1][0])-np.log(2)
        w1=np.exp(comps[1][0]-np.logaddexp(comps[0][0],comps[1][0]))
        mpre=w1[...,None]*comps[1][1]+(1-w1[...,None])*comps[0][1]
        A=s.S*(1-mpre)+0.3;B=s.S*mpre+0.3
        k1q=cnt_all[None,:,1]-k1p;k0q=cnt_all[None,:,0]-k0p
        return L+bb(k1q,k0q,A,B).sum(-1),(k1q+A)/(k1q+k0q+A+B),L,mpre
    def machine(s,hist,cutoff,grid,mtype=None):
        """returns list of hypotheses: (logprior+ll_none, ll_tau[G], pred_none, pred_tau[G])"""
        hyps=[]
        for pr in PAIRS:
            prior=(s.mainp or MAINP).get(pr,s.other)
            T,A,B,Y=arrays(hist,cutoff,pr)
            if len(T)<6: continue
            a=hist.get(cutoff-pr[0]); b=hist.get(cutoff-pr[1])
            WT=np.ones(len(T)) if not s.hl else 2.0**(-(cutoff-1-T)/s.hl)
            for model in (("x",) if s.xnor_only else ("x","f")):
                if model=="x": C=(A==B).astype(int); nc=2; mp=1-s.p4
                else: C=2*A+B; nc=4; mp=s.p4
                cnt=np.zeros((len(grid)+1,nc,2))
                for g,tau in enumerate(list(grid)+[cutoff+1]):
                    m=T<tau; np.add.at(cnt[g],(C[m],Y[m]),WT[m])
                if model=="x" and s.tps>0:
                    allc=np.concatenate([cnt,cnt[-1:]],0)  # last row: tau beyond -> none
                    llt_all,pt_all,Lpre,mpre=s.evid_x(allc[:-1],cnt[-1],mtype)
                    llt,pt=llt_all[:-1],pt_all[:-1]; lln=Lpre[-1]; pn=mpre[-1]
                else:
                    llt,pt=s.evid(cnt[:-1],cnt[-1])
                    ca=cnt[-1]; lln=bb(ca[:,1],ca[:,0],s.a0,s.a0).sum(); pn=(ca[:,1]+s.a0)/(ca.sum(1)+2*s.a0)
                if a is None or b is None:
                    # marginalize unknown lag with 50/50 over cells
                    cellw=None
                    predn=pn.mean(); predt=pt.mean(1)
                else:
                    cell=int(a>0)==int(b>0) if model=="x" else 2*(a>0)+(b>0)
                    predn=pn[int(cell)]; predt=pt[:,int(cell)]
                hyps.append((np.log(prior*mp),lln,llt,predn,predt))
        return hyps
    def predict_episode(s,rows,H):
        cutoff=rows[0].cutoff
        grid=np.arange(s.tmin,cutoff-1,s.step); G=len(grid)
        mach=[]
        for r in rows:
            hy=s.machine(H[r.machine_id],cutoff,grid,r.machine_type)
            lp=np.array([h[0] for h in hy])
            LN=np.array([h[1] for h in hy]); LT=np.array([h[2] for h in hy])
            PN=np.array([h[3] for h in hy]); PT=np.array([h[4] for h in hy])
            # machine marginal evidence per tau (over hypotheses) and none
            en=np.logaddexp.reduce(lp+LN); et=np.logaddexp.reduce(lp[:,None]+LT,axis=0)
            mach=mach+[(lp,LN,LT,PN,PT,en,et)]
        EN=sum(m[5] for m in mach); ET=sum(m[6] for m in mach)
        lpn=np.log(1-s.pchange); lpt=np.log(s.pchange/G)
        # shared hypothesis evidence vs independent
        ev_sh=np.logaddexp(EN+lpn, np.logaddexp.reduce(ET+lpt))
        ev_in=sum(np.logaddexp(m[5]+lpn, np.logaddexp.reduce(m[6]+lpt)) for m in mach)
        wsh=1/(1+np.exp(np.clip(ev_in-ev_sh-s.lsh,-50,50))) if s.lam<0 else s.lam
        out={}
        for r,(lp,LN,LT,PN,PT,en,et) in zip(rows,mach):
            preds=[]
            for mode in ("sh","in"):
                if mode=="sh": on=(EN-en)+lpn; ot=(ET-et)+lpt
                else: on=lpn; ot=np.full(G,lpt)
                Wn=lp+LN+on; Wt=lp[:,None]+LT+ot[None,:]
                m=max(Wn.max(),Wt.max()); wn=np.exp(Wn-m); wt=np.exp(Wt-m); Z=wn.sum()+wt.sum()
                preds.append(float((wn@PN+(wt*PT).sum())/Z))
            out[r.query_id]=wsh*preds[0]+(1-wsh)*preds[1]
        return out

def run(sp="validation",size_kw={},**kw):
    q,H=load(sp); m=CP2(**kw); res=[]
    for ep,g in q.groupby("episode_id"):
        rows=list(g.itertuples()); o=m.predict_episode(rows,H)
        for r in rows:
            pw=o[r.query_id]; sd=size_lag(H[r.machine_id],r.cutoff,r.machine_type,**size_kw)
            p=np.concatenate([[1-pw],pw*sd]); p=np.clip(p,1e-7,1); p/=p.sum()
            res.append([r.query_id,*p])
    return pd.DataFrame(res,columns=["query_id"]+PC)
if __name__=="__main__":
    import time;t0=time.time(); pr=run(); print(time.time()-t0); report(pr,"cp2")
    pr.to_csv("./work/v_cp2.csv",index=False)
