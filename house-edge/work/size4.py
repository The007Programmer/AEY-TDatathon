import sys; sys.path.insert(0,'work')
from cp import *
q,H=load("train")
qq=q.set_index("machine_id")
def ll(k1,n):
    if n==0: return 0
    p=(k1+.5)/(n+1); return k1*np.log(p)+(n-k1)*np.log(1-p)
cat=lambda v: 0 if v==0 else (1 if v==1 else 2)
res={}
for l in range(1,9):
  for mode in ["cat3","is1"]:
    G=0; df=0
    for r in q.itertuples():
        s=H[r.machine_id]
        wt=[t for t,v in s.items() if v>0 and s.get(t-l) is not None]
        y=np.array([s[t]==1 for t in wt]); k=np.array([cat(s[t-l]) if mode=="cat3" else int(s[t-l]==1) for t in wt])
        G+=sum(ll(y[k==c].sum(),(k==c).sum()) for c in set(k))-ll(y.sum(),len(y)); df+=len(set(k))-1
    res[(l,mode)]=(round(G,1),df)
print(res)
# time trend: first half vs second half
G=0;df=0
for r in q.itertuples():
    s=H[r.machine_id]; wt=sorted(t for t,v in s.items() if v>0); y=np.array([s[t]==1 for t in wt]); h=len(y)//2
    G+=ll(y[:h].sum(),h)+ll(y[h:].sum(),len(y)-h)-ll(y.sum(),len(y)); df+=1
print("time split",G,df)
# dependence on current-tick win state of neighbor? on episode-level 1-rate at t-1
