"""Pure CFR blueprint player (no opponent modelling)."""
from _common import *
from bot import blueprint as BP
def new_half(config, memory): memory["rng"] = np.random.default_rng()
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    return BP.act(obs, memory.setdefault("rng", np.random.default_rng()))
