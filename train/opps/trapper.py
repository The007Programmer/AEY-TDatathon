"""Slow-plays monsters (check/call until the river, then raise), bets medium hands, bluffs rarely."""
from _common import *
import random
def new_half(config, memory): memory["r"] = random.Random(5)
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    r = memory.setdefault("r", random.Random(5))
    e = eq_random(obs, 250)
    if obs.street == Street.PREFLOP:
        p = EQ.preflop_pct(obs.hole_cards)
        if p > 0.9: return Action.CHECK_CALL           # trap preflop too
        if p > 0.6: return pick(obs, [Action.RAISE_HALF]) if obs.raises_this_street == 0 else Action.CHECK_CALL
        if obs.to_call == 0: return Action.CHECK_CALL
        return Action.CHECK_CALL if p > 0.35 and obs.to_call <= 4 else Action.FOLD
    if e > 0.8:
        if obs.street == Street.RIVER: return pick(obs, [Action.RAISE_POT])
        return Action.CHECK_CALL
    if e > 0.6: return pick(obs, [Action.RAISE_HALF]) if obs.raises_this_street == 0 else Action.CHECK_CALL
    if obs.to_call == 0: return pick(obs, [Action.RAISE_HALF]) if r.random() < 0.1 else Action.CHECK_CALL
    return Action.CHECK_CALL if e > pot_odds(obs) + 0.05 else Action.FOLD
