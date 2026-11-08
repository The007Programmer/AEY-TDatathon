import numpy as np
FAM={"steady":(-0.705,-1.406,-3.892),"balanced":(0.210,-0.981,-1.792),"jackpot":(0.265,0.251,-1.099)}
MU=np.linspace(-3,3,61); BE=np.linspace(-4,4,81)
M_,B_=np.meshgrid(MU,BE,indexing="ij")
def size_lag(hist,cutoff,mtype,sd_mu=0.6,sd_b=1.2,lags=(1,2,4,3,5,6,7,8),lag_prior=None,hl=None):
    mu0,r52,r205=FAM[mtype]
    w=np.array([1.0,np.exp(r52),np.exp(r52+r205)]); big=w.sum()
    lp0=-0.5*((M_-mu0)/sd_mu)**2-0.5*(B_/sd_b)**2
    lp0-=np.logaddexp.reduce(lp0.ravel())
    if lag_prior is None: lag_prior={l:(1.0 if l in (1,2,4) else 0.15) for l in lags}
    wins=[(t,v) for t,v in hist.items() if t<cutoff and v>0]
    bigrate=(sum(v>=2 for t,v in hist.items() if t<cutoff)+1)/(len([1 for t in hist if t<cutoff])+2)
    evid=[];preds=[]
    for l in lags:
        n=np.zeros((2,2))  # [bigflag][is1]
        for t,v in wins:
            u=hist.get(t-l)
            if u is None: continue
            wt=1.0 if hl is None else 2**(-(cutoff-1-t)/hl)
            n[int(u>=2),int(v==1)]+=wt
        th0=M_; th1=M_+B_
        p0=1/(1+np.exp(th0)*big); p1=1/(1+np.exp(th1)*big)
        L=lp0+n[0,1]*np.log(p0)+n[0,0]*np.log1p(-p0)+n[1,1]*np.log(p1)+n[1,0]*np.log1p(-p1)
        Z=np.logaddexp.reduce(L.ravel()); post=np.exp(L-Z)
        u=hist.get(cutoff-l)
        if u is None: P1=bigrate*(post*p1).sum()+(1-bigrate)*(post*p0).sum()
        else: P1=(post*(p1 if u>=2 else p0)).sum()
        evid.append(Z+np.log(lag_prior[l])); preds.append(P1)
    evid=np.array(evid); ww=np.exp(evid-evid.max()); ww/=ww.sum()
    P1=float(ww@np.array(preds))
    return np.array([P1,*((1-P1)*w/big)])
