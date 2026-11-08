import sys; sys.path.insert(0,'work')
from common import *
from scipy.optimize import minimize
pr=pd.read_csv("submissions/main_v2/validation_predictions.csv").set_index("query_id")
t=pd.read_csv(P+"validation/targets.csv").set_index("query_id"); q=pd.read_csv(P+"validation/queries.csv").set_index("query_id")
idx=pr.index; lg=lambda p: np.log(p/(1-p))
X=lg(np.clip(1-pr.p_0.values,1e-6,1-1e-6)); TW=1-t.loc[idx].p_0.values
ty=q.loc[idx].machine_type.map({"steady":0,"balanced":1,"jackpot":2}).values; sg=q.loc[idx].signal.values
eps=q.loc[idx].episode_id.values; ue=np.unique(eps); rng=np.random.RandomState(0)
def f(th,m):
    a=th[ty[m]]; z=a+th[3]*X[m]+th[4]*sg[m]; return 1/(1+np.exp(-z))
def loss(th,m): return ((f(th,m)-TW[m])**2).mean()
th0=np.array([0,0,0,1,0.])
base=((1/(1+np.exp(-X))-TW)**2).mean()
cv=[]
for rep in range(5):
    perm=rng.permutation(ue); half=set(perm[:len(ue)//2])
    A=np.array([e in half for e in eps]); B=~A
    for tr,te in ((A,B),(B,A)):
        th=minimize(lambda th: loss(th,tr),th0).x
        cv.append((loss(th,te), ((1/(1+np.exp(-X[te]))-TW[te])**2).mean()))
cv=np.array(cv); print("cv recal mse",cv[:,0].mean().round(5),"base",cv[:,1].mean().round(5))
th=minimize(lambda th: loss(th,np.ones(len(X),bool)),th0).x; print("full fit",th.round(3))
