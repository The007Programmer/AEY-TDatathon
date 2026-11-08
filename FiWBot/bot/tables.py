"""
Fast, vectorizable hand evaluation for Short Deck PLO5.

Omaha hands are always exactly 2 hole cards + 3 board cards, so we store
one score for every (hole pair, board triple) combination:

    PT[pair_index * 7140 + triple_index] = dense rank (uint16, 0 = impossible)

630 pairs x 7140 triples = 4.5M entries (9 MB). Scores are "dense ranks" of
the official evaluator.py scores, so ordering and ties are identical.

PAIR_IDX[a, b] and TRIP_IDX[a, b, c] turn cards into indices for any card
order, so numpy can gather thousands of hands at once.
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
PT_PATH = os.path.join(DATA, "pt_table.npy")

N_CARDS = 36
N_PAIRS = 630
N_TRIPS = 7140

# Index arrays (all orderings filled in).
PAIR_IDX = np.full((N_CARDS, N_CARDS), -1, dtype=np.int32)
PAIRS = list(combinations(range(N_CARDS), 2))
for _i, (_a, _b) in enumerate(PAIRS):
    PAIR_IDX[_a, _b] = PAIR_IDX[_b, _a] = _i

TRIP_IDX = np.full((N_CARDS, N_CARDS, N_CARDS), -1, dtype=np.int32)
TRIPS = list(combinations(range(N_CARDS), 3))
for _i, (_a, _b, _c) in enumerate(TRIPS):
    for _p in ((_a, _b, _c), (_a, _c, _b), (_b, _a, _c), (_b, _c, _a), (_c, _a, _b), (_c, _b, _a)):
        TRIP_IDX[_p] = _i

# Positions of the 10 pairs among 5 cards and the 10 triples among 5 cards.
C52 = np.array(list(combinations(range(5), 2)), dtype=np.int64)   # (10, 2)
C53 = np.array(list(combinations(range(5), 3)), dtype=np.int64)   # (10, 3)


def build_pt_table() -> np.ndarray:
    """Build the pair x triple table from the official evaluator (offline, ~10 s)."""
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from evaluator import get_table

    table = get_table()
    distinct = sorted(set(table.values()))
    dense = {s: i + 1 for i, s in enumerate(distinct)}       # 0 reserved for "impossible"
    assert len(distinct) < 65000
    pair_masks = [(1 << a) | (1 << b) for a, b in PAIRS]
    trip_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in TRIPS]
    pt = np.zeros(N_PAIRS * N_TRIPS, dtype=np.uint16)
    for pi, pm in enumerate(pair_masks):
        base = pi * N_TRIPS
        row = pt[base:base + N_TRIPS]
        for ti, tm in enumerate(trip_masks):
            if pm & tm == 0:
                row[ti] = dense[table[pm | tm]]
    return pt


def load_pt_table() -> np.ndarray:
    if os.path.exists(PT_PATH):
        return np.load(PT_PATH)
    pt = build_pt_table()
    try:
        os.makedirs(DATA, exist_ok=True)
        np.save(PT_PATH, pt)
    except OSError:
        pass          # read-only filesystem: just keep it in memory
    return pt


PT = load_pt_table()


def hole_pair_idx(hole) -> np.ndarray:
    """(10,) pair indices for a 5-card hole (flat offsets into PT rows)."""
    h = np.asarray(hole, dtype=np.int64)
    return PAIR_IDX[h[C52[:, 0]], h[C52[:, 1]]].astype(np.int64) * N_TRIPS


def board_trip_idx(board) -> np.ndarray:
    """(10,) triple indices for a 5-card board."""
    b = np.asarray(board, dtype=np.int64)
    return TRIP_IDX[b[C53[:, 0]], b[C53[:, 1]], b[C53[:, 2]]].astype(np.int64)


def omaha_rank(hole, board) -> int:
    """Dense rank of the best 2+3 hand (same order as evaluator.omaha_score)."""
    return int(PT[hole_pair_idx(hole)[:, None] + board_trip_idx(board)[None, :]].max())


def batch_pair_idx(holes: np.ndarray) -> np.ndarray:
    """holes (K,5) -> (K,10) flat pair offsets."""
    return PAIR_IDX[holes[:, C52[:, 0]], holes[:, C52[:, 1]]].astype(np.int64) * N_TRIPS


def batch_trip_idx(boards: np.ndarray) -> np.ndarray:
    """boards (K,5) -> (K,10) triple indices."""
    return TRIP_IDX[boards[:, C53[:, 0]], boards[:, C53[:, 1]], boards[:, C53[:, 2]]].astype(np.int64)


def batch_rank(pair_off: np.ndarray, trip_idx: np.ndarray) -> np.ndarray:
    """pair_off (K,10) or (10,), trip_idx (K,10) -> (K,) best ranks."""
    if pair_off.ndim == 1:
        idx = pair_off[None, :, None] + trip_idx[:, None, :]
    else:
        idx = pair_off[:, :, None] + trip_idx[:, None, :]
    return PT[idx].reshape(idx.shape[0], -1).max(axis=1)


if __name__ == "__main__":
    import time
    t = time.time()
    pt = build_pt_table()
    os.makedirs(DATA, exist_ok=True)
    np.save(PT_PATH, pt)
    print("built", pt.shape, "max rank", pt.max(), f"{time.time() - t:.1f}s")
