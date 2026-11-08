"""
Offline: preflop equity of every 5-card hand vs a random hand (Monte Carlo).

Hands are grouped into suit-isomorphic classes (relabelling suits doesn't
change equity), each class is simulated once, and the result is written to
data/preflop_eq.npy, indexed by the 5-card combinatorial index (376,992).

Run:  .venv/bin/python -m train.build_preflop [trials_per_class]
"""

from __future__ import annotations

import os
import sys
import time
from itertools import combinations, permutations
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import tables as T            # noqa: E402
from bot.equity import combo5_index    # noqa: E402

OUT = os.path.join(T.DATA, "preflop_eq.npy")


def canonical_keys(hands: np.ndarray) -> np.ndarray:
    """(N,5) sorted hands -> (N,) int64 canonical key under suit relabelling."""
    best = None
    for perm in permutations(range(4)):
        p = np.array(perm, dtype=np.int64)
        mapped = (hands & ~3) | p[hands & 3]
        mapped.sort(axis=1)
        key = np.zeros(len(hands), dtype=np.int64)
        for j in range(5):
            key = key * 36 + mapped[:, j]
        best = key if best is None else np.minimum(best, key)
    return best


def class_equity(args):
    hand, trials, seed = args
    rng = np.random.default_rng(seed)
    hand = np.asarray(hand, dtype=np.int64)
    rest = np.setdiff1d(np.arange(36), hand)                 # 31 cards
    my_pairs = T.hole_pair_idx(hand)
    wins = 0.0
    done = 0
    while done < trials:
        k = min(4000, trials - done)
        perm = np.argsort(rng.random((k, len(rest))), axis=1)[:, :10]
        draw = rest[perm]
        opp, board = draw[:, :5], draw[:, 5:]
        trips = T.batch_trip_idx(board)
        mine = T.batch_rank(my_pairs, trips)
        theirs = T.batch_rank(T.batch_pair_idx(opp), trips)
        wins += (mine > theirs).sum() + 0.5 * (mine == theirs).sum()
        done += k
    return wins / trials


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.time()
    hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    keys = canonical_keys(hands)
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    print(f"{len(hands)} hands, {len(uniq)} classes ({time.time() - t0:.1f}s)")
    reps = hands[first]
    jobs = [(tuple(r), trials, i) for i, r in enumerate(reps)]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        eqs = np.array(pool.map(class_equity, jobs, chunksize=64), dtype=np.float32)
    per_hand = eqs[inverse]
    out = np.zeros(len(hands), dtype=np.float32)
    out[combo5_index(hands)] = per_hand
    np.save(OUT, out)
    print(f"saved {OUT}: min {out.min():.3f} mean {out.mean():.3f} max {out.max():.3f} "
          f"({time.time() - t0:.0f}s)")
    order = np.argsort(eqs)
    from evaluator import cards_to_str
    print("best :", [(cards_to_str(reps[i]), round(float(eqs[i]), 3)) for i in order[-5:]])
    print("worst:", [(cards_to_str(reps[i]), round(float(eqs[i]), 3)) for i in order[:5]])


if __name__ == "__main__":
    main()
