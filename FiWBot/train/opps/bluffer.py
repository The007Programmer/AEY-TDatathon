"""Hyper-aggressive bluffer: bets whenever checked to, raises randomly, folds weak to resistance."""
from _common import *
import random
def new_half(config, memory): memory["r"] = random.Random(11)
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    r = memory.setdefault("r", random.Random(11))
    e = eq_random(obs, 200)
    if obs.to_call == 0:
        return pick(obs, [Action.RAISE_POT if r.random() < 0.5 else Action.RAISE_HALF]) if r.random() < 0.8 or e > 0.6 else Action.CHECK_CALL
    if e > 0.7 or r.random() < 0.2:
        return pick(obs, [Action.RAISE_POT])
    return Action.CHECK_CALL if e > pot_odds(obs) else Action.FOLD
