"""Opponent-model frequencies our bot observes against each opponent (blueprint-only play)."""
import sys, numpy as np
from train.arena import load_bot, Seat
from game import Game, make_deal, button_for_deal, AgentConfig
from bot import opponent as OM
me = load_bot("agent.py?bp_mode=4&bp_margin=1000000.0")
for opp in sys.argv[1:]:
    m2 = load_bot(opp); s1, s2 = Seat(me), Seat(m2)
    for s in (s1, s2): s.start(AgentConfig())
    snaps = {}
    for d in range(1000):
        g = Game(make_deal("prof", d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            st = g.to_act; g.apply((s1, s2)[st].act(g.observation(st)))
        for st in (0, 1): (s1, s2)[st].hand_end(g.hand_result(st))
        if d + 1 in (100, 300, 1000):
            m = s1.memory["model"]
            snaps[d + 1] = {k: {a: round(OM.freq(m, k, a), 2) for a in c} for k, c in m["counts"].items()}
    print(opp.split("/")[-1])
    for k in snaps[1000]:
        print("   ", k, " | ".join(f"{n}:{snaps[n][k]}" for n in (100, 300, 1000)))
