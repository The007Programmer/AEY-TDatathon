import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]
# For drift/shared machines: with the chosen pair, look at the cell win rate over time (windows)
import pandas as pd, numpy as np, itertools
P=str(ROOT/'kit/public')+'/'; sp="train"
q=pd.read_csv(P+sp+"/queries.csv"); c=pd.read_csv(P+sp+"/conditions.csv"); d=q.merge(c,on="query_id")
h=pd.read_csv(P+sp+"/history.csv")
H={k:dict(zip(g.tick,g.payout)) for k,g in h.groupby("machine_id")}
def cells(s,pr):
    out=[]
    for tk,v in sorted(s.items()):
        ctx=[s.get(tk-l) for l in pr]
        if None in ctx: continue
        out.append((tk,tuple(int(x>0) for x in ctx),int(v>0)))
    return out
for cond in ["drift","shared_change"]:
    sub=d[d.condition==cond]
    for ep in sub.episode_id.unique()[:2]:
        print("==",cond,ep)
        for r in sub[sub.episode_id==ep].itertuples():
            s=H[r.machine_id]
            # best pair overall by abs diff
            best=None
            for pr in [(2,5),(4,7),(1,3)]:
                cl=cells(s,pr)
                line=[]
                for w0 in range(0,r.cutoff,32):
                    seg=[x for x in cl if w0<=x[0]<w0+32]
                    rates=[]
                    for key in [(0,0),(0,1),(1,0),(1,1)]:
                        z=[x[2] for x in seg if x[1]==key]
                        rates.append(f"{np.mean(z):.1f}" if z else " - ")
                    line.append("/".join(rates))
                print(r.machine_type[:3],pr," | ".join(line))
