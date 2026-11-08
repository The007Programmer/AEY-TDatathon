"""Only plays strong hands."""
from _common import *
def new_half(config, memory): pass
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    e = eq_random(obs, 200)
    if obs.street == Street.PREFLOP:
        p = EQ.preflop_pct(obs.hole_cards)
        if p > 0.85: return pick(obs, [Action.RAISE_POT])
        if p > 0.65: return Action.CHECK_CALL
        return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    if e > 0.72: return pick(obs, [Action.RAISE_POT])
    if obs.to_call == 0: return Action.CHECK_CALL
    return Action.CHECK_CALL if e > 0.6 else Action.FOLD
