"""
Monte Carlo equity with a weighted opponent range (numpy, single thread).

One call to `simulate` samples K opponent hands + board runouts consistent
with the cards we can see (blockers handled automatically), and returns
everything the strategy needs per sample: who wins, and the opponent's
strength features (preflop percentile, made-hand percentile per street,
draws). The strategy then reweights those samples to model the opponent's
range, which is far cheaper than re-simulating.
"""

from __future__ import annotations

import os
from math import comb

import numpy as np

from . import tables as T

# --- 5-card combinatorial index (for the preflop table) ----------------------
_BIN = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)


def combo5_index(hands: np.ndarray) -> np.ndarray:
    """(N,5) hands (any order) -> (N,) index in 0..376,991."""
    h = np.sort(np.asarray(hands, dtype=np.int64), axis=1)
    return (_BIN[h[:, 0], 1] + _BIN[h[:, 1], 2] + _BIN[h[:, 2], 3]
            + _BIN[h[:, 3], 4] + _BIN[h[:, 4], 5])


PF_PATH = os.path.join(T.DATA, "preflop_eq.npy")
if os.path.exists(PF_PATH):
    PF_EQ = np.load(PF_PATH)
    _order = np.argsort(PF_EQ, kind="stable")
    PF_PCT = np.empty_like(PF_EQ)
    PF_PCT[_order] = (np.arange(len(PF_EQ), dtype=np.float32) + 0.5) / len(PF_EQ)
else:                                  # only while building the table offline
    PF_EQ = PF_PCT = None


def preflop_pct(hole) -> float:
    return float(PF_PCT[combo5_index(np.asarray(hole)[None, :])[0]])


def preflop_eq(hole) -> float:
    return float(PF_EQ[combo5_index(np.asarray(hole)[None, :])[0]])


# --- simulation ----------------------------------------------------------------

class Sim:
    """Result of one simulation (all arrays length K)."""
    __slots__ = ("outcome", "pf_pct", "street_pct", "street_rank", "draw", "k", "my_made")


def _made_rank(pair_off: np.ndarray, board_cards) -> np.ndarray:
    """Best 2+3 rank using the visible board only (3, 4 or 5 cards)."""
    b = list(board_cards)
    trips = np.array([T.TRIP_IDX[x, y, z] for i, x in enumerate(b) for j, y in enumerate(b)
                      if j > i for z in b[j + 1:]], dtype=np.int64)
    idx = pair_off[:, :, None] + trips[None, None, :]
    return T.PT[idx].reshape(len(pair_off), -1).max(axis=1)


def _pct(x: np.ndarray) -> np.ndarray:
    """Rank-transform to (0,1) with ties averaged."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x))
    # average ties
    sx = x[order]
    _, start, counts = np.unique(sx, return_index=True, return_counts=True)
    avg = start + (counts - 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return (ranks + 0.5) / len(x)


def simulate(rng: np.random.Generator, hole, board, k: int, need_streets=()) -> Sim:
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    dead = np.zeros(36, dtype=bool)
    dead[hole] = True
    dead[board] = True
    rest = np.flatnonzero(~dead)
    n_run = 5 - len(board)
    need = 5 + n_run
    draw = rest[np.argsort(rng.random((k, len(rest))), axis=1)[:, :need]]
    opp = draw[:, :5]
    if n_run:
        full = np.concatenate([np.broadcast_to(board, (k, len(board))), draw[:, 5:]], axis=1)
        trips = T.batch_trip_idx(full)
        mine = T.batch_rank(T.hole_pair_idx(hole), trips)
    else:
        trips = np.broadcast_to(T.board_trip_idx(board), (k, 10))
        mine = np.full(k, T.omaha_rank(hole, board))
    opp_pairs = T.batch_pair_idx(opp)
    theirs = T.batch_rank(opp_pairs, trips)

    s = Sim()
    s.k = k
    s.outcome = (mine > theirs) + 0.5 * (mine == theirs)
    s.pf_pct = PF_PCT[combo5_index(opp)].astype(np.float64)
    s.street_pct = {}
    s.street_rank = {}
    s.draw = {}
    s.my_made = None
    for n_board in need_streets:             # 3, 4, 5 = flop, turn, river strengths
        if n_board > len(board) or n_board < 3:
            continue
        vis = board[:n_board]
        rk = _made_rank(opp_pairs, vis)
        s.street_rank[n_board] = rk
        s.street_pct[n_board] = _pct(rk)
        if n_board == len(board) and n_board < 5:
            mine_now = _made_rank(T.hole_pair_idx(hole)[None, :], vis)[0]
            s.my_made = float((rk < mine_now).mean() + 0.5 * (rk == mine_now).mean())
        if n_board < 5:
            suits_board = np.bincount(vis & 3, minlength=4)
            hole_suits = np.stack([(opp & 3) == su for su in range(4)], axis=2).sum(axis=1)
            fd = ((hole_suits >= 2) & (suits_board[None, :] == 2)).any(axis=1)
            s.draw[n_board] = fd
    return s


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    sw = w.sum()
    if sw <= 1e-12:
        return float(x.mean())
    return float((x * w).sum() / sw)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float(np.quantile(x, q))
    i = int(np.searchsorted(cw, q * cw[-1]))
    return float(x[order[min(i, len(x) - 1)]])


def ehs_made(rng: np.random.Generator, hole, board, k: int) -> tuple[float, float]:
    """(equity vs a random hand, made-hand percentile on the visible board).

    The second number separates made hands (made >= ehs) from draws
    (ehs > made): two hands with the same equity play very differently.
    """
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    sim = simulate(rng, hole, board, k)
    ehs = float(sim.outcome.mean())
    if len(board) >= 5:
        return ehs, ehs
    dead = np.zeros(36, dtype=bool)
    dead[hole] = True
    dead[board] = True
    rest = np.flatnonzero(~dead)
    opp = rest[np.argsort(rng.random((k, len(rest))), axis=1)[:, :5]]
    theirs = _made_rank(T.batch_pair_idx(opp), board)
    mine = _made_rank(T.hole_pair_idx(hole)[None, :], board)[0]
    made = float((theirs < mine).mean() + 0.5 * (theirs == mine).mean())
    return ehs, made
