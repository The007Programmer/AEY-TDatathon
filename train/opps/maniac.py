"""Loose-aggressive: raises a lot, calls a lot."""
from _common import *
def new_half(config, memory): pass
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    e = eq_random(obs, 200)
    if e > 0.45: return pick(obs, [Action.RAISE_POT, Action.RAISE_HALF])
    if obs.to_call == 0: return pick(obs, [Action.RAISE_HALF]) if e > 0.35 else Action.CHECK_CALL
    return Action.CHECK_CALL if e > 0.3 else Action.FOLD
