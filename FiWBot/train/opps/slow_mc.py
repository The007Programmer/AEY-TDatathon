"""Pure-Python MC with the official evaluator: decent play, but burns its time bank."""
from _common import *
import random
from evaluator import omaha_score
def new_half(config, memory): memory["r"] = random.Random(3)
def on_hand_end(result, memory): pass
def choose_action(obs, memory):
    r = memory.setdefault("r", random.Random(3))
    dead = set(obs.hole_cards) | set(obs.board)
    deck = [c for c in range(36) if c not in dead]
    wins = 0.0; n = 120
    for _ in range(n):
        s = r.sample(deck, 5 + 5 - len(obs.board))
        opp, board = s[:5], list(obs.board) + s[5:]
        a, b = omaha_score(obs.hole_cards, board), omaha_score(opp, board)
        wins += 1 if a > b else 0.5 if a == b else 0
    e = wins / n
    if e > 0.68: return pick(obs, [Action.RAISE_POT])
    if e > 0.58: return pick(obs, [Action.RAISE_HALF])
    if obs.to_call == 0: return Action.CHECK_CALL
    return Action.CHECK_CALL if e > pot_odds(obs) + 0.04 else Action.FOLD
