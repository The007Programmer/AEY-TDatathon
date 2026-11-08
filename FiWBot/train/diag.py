"""Where do the chips go? Breakdown of agent1's results by preflop line."""
import sys, collections
from train.arena import load_bot, Seat
from game import Game, make_deal, button_for_deal, AgentConfig, Action
def line(hist, a1_seat):
    out=[]
    for r in hist:
        who = "H" if r.seat==a1_seat else "V"
        out.append(f"{who}{'FCHP'[int(r.action)]}{'pftr'[int(r.street)]}")
    return out
def run(spec1, spec2, seed=5, deals=1000):
    m1,m2=load_bot(spec1),load_bot(spec2)
    agg=collections.defaultdict(lambda:[0,0.0]); agg2=collections.defaultdict(lambda:[0,0.0])
    for half in (1,2):
        s1,s2=Seat(m1),Seat(m2); seats=(s1,s2) if half==1 else (s2,s1); a1=0 if half==1 else 1
        for s in seats: s.start(AgentConfig())
        for d in range(deals):
            g=Game(make_deal(str(seed),d),button_for_deal(d),hand_index=d)
            while not g.is_terminal:
                st=g.to_act; g.apply(seats[st].act(g.observation(st)))
            for st in (0,1): seats[st].hand_end(g.hand_result(st))
            res=g.result(); L=line(res.history,a1)
            pre=" ".join(x[:2] for x in L if x[2]=="p")
            pos="BTN" if g.button_seat==a1 else "BB"
            k=(pos,pre[:20]); agg[k][0]+=1; agg[k][1]+=res.nets[a1]/2
            k2=(pos,"SD" if res.went_to_showdown else "noSD", "post" if any(x[2]!="p" for x in L) else "pre")
            agg2[k2][0]+=1; agg2[k2][1]+=res.nets[a1]/2
    for k,(n,v) in sorted(agg.items(), key=lambda kv: kv[1][1])[:12]: print(f"{str(k):40s} n={n:5d} total={v:+8.1f}bb avg={v/n:+.2f}")
    print("...")
    for k,(n,v) in sorted(agg.items(), key=lambda kv: kv[1][1])[-6:]: print(f"{str(k):40s} n={n:5d} total={v:+8.1f}bb avg={v/n:+.2f}")
    print()
    for k,(n,v) in sorted(agg2.items()): print(f"{str(k):40s} n={n:5d} total={v:+8.1f}bb avg={v/n:+.2f}")
if __name__=="__main__": run(sys.argv[1], sys.argv[2])
