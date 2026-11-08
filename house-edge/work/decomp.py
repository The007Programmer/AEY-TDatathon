import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'work'))
from cp import *
pr=run()
t=pd.read_csv(P+"validation/targets.csv").set_index("query_id").loc[pr.query_id]
P_=pr[PC].values; T_=t[PC].values
# oracle win prob, our size
A=P_.copy(); w=1-T_[:,0]; A[:,1:]=P_[:,1:]/P_[:,1:].sum(1,keepdims=True)*w[:,None]; A[:,0]=1-w
B=P_.copy(); w=1-P_[:,0]; B[:,1:]=T_[:,1:]/T_[:,1:].sum(1,keepdims=True)*w[:,None]; B[:,0]=1-w
for nm,X in [("ours",P_),("oracle-win",A),("oracle-size",B)]:
    report(pd.DataFrame(np.c_[pr.query_id.values,X],columns=["query_id"]+PC).astype({c:float for c in PC}),nm)
