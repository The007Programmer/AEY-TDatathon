"""
Offline: sample random deals and compute each player's hand-strength
feature on every street (for CFR bucketing) plus the showdown outcome.

Per deal (player 0 = button, player 1 = big blind):
    feat[:, p, 0]  preflop percentile (table)
    feat[:, p, 1]  flop  equity vs a random hand (MC)
    feat[:, p, 2]  turn  equity vs a random hand (MC)
    feat[:, p, 3]  river equity vs a random hand (MC over opponent hands)
    out[:]         showdown result for the button: 1 win, 0.5 split, 0 loss

Run:  .venv/bin/python -m train.gen_deals N_DEALS OUT.npz
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import equity as EQ   # noqa: E402
from bot import tables as T    # noqa: E402

K_FEAT = int(os.environ.get("K_FEAT", "192"))


def ehs(rng, hole, board, k=K_FEAT) -> float:
    return float(EQ.simulate(rng, hole, board, k).outcome.mean())


def chunk(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    feat = np.zeros((n, 2, 4), dtype=np.float16)
    out = np.zeros(n, dtype=np.float16)
    for i in range(n):
        cards = rng.permutation(36)[:15]
        h0, h1, board = cards[:5], cards[5:10], cards[10:]
        for p, h in ((0, h0), (1, h1)):
            feat[i, p, 0] = EQ.preflop_pct(h)
            feat[i, p, 1] = ehs(rng, h, board[:3])
            feat[i, p, 2] = ehs(rng, h, board[:4])
            feat[i, p, 3] = ehs(rng, h, board)
        r0, r1 = T.omaha_rank(h0, board), T.omaha_rank(h1, board)
        out[i] = 1.0 if r0 > r1 else 0.5 if r0 == r1 else 0.0
    return feat, out


def main():
    n = int(sys.argv[1])
    path = sys.argv[2]
    per = 2000
    jobs = [(min(per, n - s), 1000 + s // per) for s in range(0, n, per)]
    t = time.time()
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        parts = pool.map(chunk, jobs, chunksize=1)
    feat = np.concatenate([p[0] for p in parts])
    out = np.concatenate([p[1] for p in parts])
    np.savez(path, feat=feat, out=out)
    print(f"{n} deals in {time.time() - t:.0f}s -> {path}")


if __name__ == "__main__":
    main()
