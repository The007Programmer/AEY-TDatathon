"""
Opponent model: action frequencies per situation, a "how much do their
actions mean" reliability score from showdowns, and dead-bot detection.

Situations (keys):
    ("pre", "sb")      small blind's first preflop decision (fold / call=limp / raise)
    ("pre", "facing")  preflop, facing a bet or raise
    ("pre", "open")    preflop, nothing to call (big blind after a limp)
    ("post", "facing") flop/turn/river, facing a bet or raise
    ("post", "open")   flop/turn/river, nothing to call
"""

from __future__ import annotations

from collections import deque

from game import Action, Street

PRIORS = {
    ("pre", "sb"): {"fold": 1.5, "call": 3.0, "raise": 5.5},
    ("pre", "facing"): {"fold": 3.0, "call": 4.5, "raise": 2.5},
    ("pre", "open"): {"check": 6.0, "raise": 4.0},
    ("post", "facing"): {"fold": 4.0, "call": 4.5, "raise": 1.5},
    ("post", "open"): {"check": 5.5, "raise": 4.5},
}
DECAY = 0.996           # per hand: effective memory ~250 hands, so we track changes
LAMBDA_PRIOR = 0.6      # prior "their bets mean strength" reliability
LAMBDA_PRIOR_N = 6.0


def new_model() -> dict:
    return {
        "counts": {k: dict(v) for k, v in PRIORS.items()},
        "recent_facing": deque(maxlen=12),     # opp responses when facing a bet
        "recent_raise": deque(maxlen=30),      # 1 if opp raised in any decision
        "cal": {"pre": [0.0, 0.0], "post": [0.0, 0.0]},  # [sum of strength - 0.5 excess, n]
        "hands": 0,
    }


def group_of(street: int) -> str:
    return "pre" if street == Street.PREFLOP else "post"


def record(model: dict, key: tuple, kind: str) -> None:
    c = model["counts"][key]
    if kind not in c:
        kind = "call" if kind == "check" else kind
    if kind in c:
        c[kind] += 1.0
    if key[1] in ("facing", "sb"):
        model["recent_facing"].append(kind)
    model["recent_raise"].append(1 if kind == "raise" else 0)


def decay(model: dict) -> None:
    model["hands"] += 1
    for key, c in model["counts"].items():
        pr = PRIORS[key]
        for a in c:
            c[a] = pr[a] + (c[a] - pr[a]) * DECAY


def freq(model: dict, key: tuple, kind: str) -> float:
    c = model["counts"][key]
    tot = sum(c.values())
    return c.get(kind, 0.0) / tot if tot > 0 else 0.0


def is_zombie(model: dict) -> bool:
    """Opponent auto check/folds (time bank empty, forfeited, or a nit bot)."""
    rf = model["recent_facing"]
    if len(rf) < 8:
        return False
    return all(k == "fold" for k in rf) and sum(model["recent_raise"]) == 0


def fold_freq(model: dict, group: str) -> float:
    if is_zombie(model):
        return 0.97
    return freq(model, (group, "facing"), "fold")


def reliability(model: dict, group: str) -> float:
    """0 = their actions say nothing about their cards, 1 = fully strength-based."""
    s, n = model["cal"][group]
    # s accumulates (strength - 0.5) / 0.25 for hands they bet with
    est = (LAMBDA_PRIOR * LAMBDA_PRIOR_N + s) / (LAMBDA_PRIOR_N + n)
    return min(1.0, max(0.05, est))


def calibrate(model: dict, group: str, strength: float) -> None:
    """Strength percentile of a hand the opponent bet/raised with (seen at showdown)."""
    model["cal"][group][0] += (strength - 0.5) / 0.3
    model["cal"][group][1] += 1.0


# --- reading the action history -------------------------------------------

def parse_history(obs) -> list:
    """Replay obs.action_history. Returns a list of events
    (street, is_me, key, kind) with kind in fold/call/check/raise."""
    events = []
    me_bet, opp_bet = (1, 2) if obs.is_button else (2, 1)
    street = Street.PREFLOP
    first = True
    for rec in obs.action_history:
        if rec.street != street:
            street = rec.street
            me_bet = opp_bet = 0
        actor_bet, other_bet = (me_bet, opp_bet) if rec.is_me else (opp_bet, me_bet)
        facing = actor_bet < other_bet
        g = group_of(street)
        if street == Street.PREFLOP and first:
            key = ("pre", "sb")
        else:
            key = (g, "facing" if facing else "open")
        first = False
        a = rec.action
        if a == Action.FOLD:
            kind = "fold"
        elif a == Action.CHECK_CALL:
            kind = "call" if facing else "check"
        else:
            kind = "raise"
        events.append((int(street), bool(rec.is_me), key, kind))
        if rec.is_me:
            me_bet = rec.street_total
        else:
            opp_bet = rec.street_total
    return events


# Normal ranges, measured from the CFR blueprint playing itself (train/profile.py).
# An opponent outside them is far from balanced, so exploiting it beats
# playing the blueprint; inside them, the blueprint is the safer choice.
BALANCED = {
    (("pre", "facing"), "fold"): (0.15, 0.48),
    (("post", "facing"), "fold"): (0.25, 0.62),
    (("pre", "sb"), "fold"): (0.0, 0.45),
    (("post", "open"), "raise"): (0.15, 0.55),
    (("post", "facing"), "raise"): (0.0, 0.30),
    (("pre", "facing"), "raise"): (0.05, 0.42),
}
MIN_HANDS = 60


def exploit_reasons(model: dict) -> list:
    """Why this opponent looks exploitable (empty list = looks balanced)."""
    if is_zombie(model):
        return ["zombie"]
    if model["hands"] < MIN_HANDS:
        return []
    out = []
    for (key, kind), (lo, hi) in BALANCED.items():
        f = freq(model, key, kind)
        if f < lo or f > hi:
            out.append(f"{key[0]}-{key[1]}-{kind}={f:.2f}")
    for g in ("pre", "post"):
        if model["cal"][g][1] >= 8 and reliability(model, g) < 0.3:
            out.append(f"{g}-unreliable")
    return out
