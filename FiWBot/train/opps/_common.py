"""Shared helpers for local sparring bots (not shipped)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from bot import equity as EQ
from game import Action, Street

_rng = np.random.default_rng(12345)

def eq_random(obs, k=300):
    s = EQ.simulate(_rng, obs.hole_cards, obs.board, k)
    return float(s.outcome.mean())

def pick(obs, want):
    """first legal of the wanted actions, else check/call"""
    for a in want:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL

def pot_odds(obs):
    return obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
