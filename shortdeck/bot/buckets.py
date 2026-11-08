"""
Hand-strength buckets shared by CFR training and live play (must match).

    preflop : NB bins of the preflop percentile
    flop/turn: N_EHS bins of equity-vs-random  x  3 bins of (equity - made percentile)
               i.e. the same equity split into "made hand" / "mixed" / "drawing hand"
    river   : NB bins of equity-vs-random
"""

from __future__ import annotations

import numpy as np

NB = 48
N_EHS = 16
N_D = 3


def make_edges(feat: np.ndarray, made: np.ndarray) -> dict:
    """feat/made (N,2,4) -> bucket edges (quantiles, so buckets are equally full)."""
    e = {}
    e["pf"] = (np.arange(1, NB) / NB).astype(np.float32)
    for s, name in ((1, "flop"), (2, "turn")):
        x = feat[:, :, s].astype(np.float32).ravel()
        d = x - made[:, :, s].astype(np.float32).ravel()
        e[name + "_ehs"] = np.quantile(x, np.arange(1, N_EHS) / N_EHS).astype(np.float32)
        e[name + "_d"] = np.quantile(d, np.arange(1, N_D) / N_D).astype(np.float32)
    x = feat[:, :, 3].astype(np.float32).ravel()
    e["river"] = np.quantile(x, np.arange(1, NB) / NB).astype(np.float32)
    return e


def bucket(street: int, ehs, made, e: dict):
    """Vectorised: street 0..3, ehs/made scalars or arrays -> bucket ids in 0..NB-1."""
    ehs = np.asarray(ehs, dtype=np.float32)
    if street == 0:
        return np.searchsorted(e["pf"], ehs, side="right")
    if street == 3:
        return np.searchsorted(e["river"], ehs, side="right")
    name = "flop" if street == 1 else "turn"
    be = np.searchsorted(e[name + "_ehs"], ehs, side="right")
    bd = np.searchsorted(e[name + "_d"], ehs - np.asarray(made, dtype=np.float32), side="right")
    return be * N_D + bd


def bucketize(feat: np.ndarray, made: np.ndarray, e: dict) -> np.ndarray:
    b = np.zeros(feat.shape, dtype=np.int16)
    for s in range(4):
        b[:, :, s] = bucket(s, feat[:, :, s], made[:, :, s], e)
    return b
