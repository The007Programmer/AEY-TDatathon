"""Tight-aggressive rules bot: preflop percentile, postflop equity tiers, bluffs a bit."""
from _common import *
import random
def new_half(config, memory): memory["r"] = random.Random(7)
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    r = memory.setdefault("r", random.Random(7))
    if obs.street == Street.PREFLOP:
        p = EQ.preflop_pct(obs.hole_cards)
        if p > 0.80: return pick(obs, [Action.RAISE_POT])
        if p > 0.55 and obs.raises_this_street < 2: return pick(obs, [Action.RAISE_HALF])
        if p > 0.35 and obs.to_call <= 4: return Action.CHECK_CALL
        if obs.to_call == 0: return Action.CHECK_CALL
        return Action.FOLD if p < 0.6 else Action.CHECK_CALL
    e = eq_random(obs, 250)
    if e > 0.78: return pick(obs, [Action.RAISE_POT])
    if e > 0.65: return pick(obs, [Action.RAISE_HALF]) if obs.raises_this_street < 2 else Action.CHECK_CALL
    if obs.to_call == 0:
        return pick(obs, [Action.RAISE_HALF]) if r.random() < 0.25 else Action.CHECK_CALL
    return Action.CHECK_CALL if e > pot_odds(obs) + 0.1 else Action.FOLD
