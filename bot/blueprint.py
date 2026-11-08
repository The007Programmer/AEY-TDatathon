"""
Play the CFR blueprint (data/blueprint.npz).

The live action history maps exactly onto a tree node (the tree is built
from game.py's own rules), our hand strength maps onto a bucket, and the
blueprint gives a probability for each legal action.
"""

from __future__ import annotations

import os

import numpy as np

from game import Action, Street

from . import equity as EQ
from .tables import DATA
from .tree import Tree

BP_PATH = os.path.join(DATA, "blueprint.npz")
AVAILABLE = os.path.exists(BP_PATH)
if AVAILABLE:
    _bp = np.load(BP_PATH)
    STRAT = _bp["strat"].astype(np.float32)
    EDGES = _bp["edges"]
    TREE = Tree()
    INDEX = TREE.index
K_RUNTIME = 192


def node_of(obs):
    key = tuple(int(r.action) for r in obs.action_history)
    return INDEX.get(key)


def strength(obs, rng, k: int = K_RUNTIME) -> float:
    if obs.street == Street.PREFLOP:
        return EQ.preflop_pct(obs.hole_cards)
    return float(EQ.simulate(rng, obs.hole_cards, obs.board, k).outcome.mean())


def probs(obs, rng, feat: float | None = None):
    """{action: probability} over legal actions, or None if unknown node."""
    n = node_of(obs)
    if n is None:
        return None
    if feat is None:
        feat = strength(obs, rng)
    b = int(np.searchsorted(EDGES[int(obs.street)], feat, side="right"))
    row = STRAT[n, b]
    out = {a: float(row[int(a)]) for a in obs.legal_actions}
    tot = sum(out.values())
    if tot <= 0:
        return {a: 1.0 / len(out) for a in out}
    return {a: v / tot for a, v in out.items()}


def act(obs, rng) -> Action:
    p = probs(obs, rng)
    if p is None:
        return Action.CHECK_CALL
    acts = list(p.keys())
    return acts[int(rng.choice(len(acts), p=np.array([p[a] for a in acts])))]
