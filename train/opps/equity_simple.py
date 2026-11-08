"""Typical hackathon bot: MC equity vs a random hand + pot odds."""
from _common import *
def new_half(config, memory): pass
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    e = eq_random(obs)
    hi = 0.60 if obs.street == Street.PREFLOP else 0.70
    if e > hi:
        return pick(obs, [Action.RAISE_POT])
    if e > hi - 0.08:
        return pick(obs, [Action.RAISE_HALF])
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if e > pot_odds(obs) + 0.05:
        return Action.CHECK_CALL
    return Action.FOLD
