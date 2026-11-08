"""
Offline: chance-sampled CFR+ on the exact betting tree with strength buckets.

Infoset = (tree node, actor's strength bucket on the current street).
Each batch takes B pre-computed deals (train/gen_deals.py) and walks the
whole public tree once with numpy vectors of length B: every deal gets its
own reach probabilities and utilities, and regrets/strategy sums are
scatter-added per bucket. Several worker processes share the regret table
(lock-free "hogwild" updates).

Output: data/blueprint.npz with
    strat  (n_nodes, NB, 4) float16  average strategy (rows of non-decision nodes are 0)
    edges  (4, NB-1)        float32  bucket edges per street (street 0 = preflop percentile)

Run:  .venv/bin/python -m train.cfr train/data/deals_4m.npz --epochs 3 --batch 16384 --workers 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Process, Value, shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot.tree import Tree  # noqa: E402

NB = 32
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "blueprint.npz")


def make_edges(feat: np.ndarray) -> np.ndarray:
    edges = np.zeros((4, NB - 1), dtype=np.float32)
    qs = np.arange(1, NB) / NB
    edges[0] = qs                                       # preflop percentile is uniform already
    for s in (1, 2, 3):
        x = feat[:, :, s].astype(np.float32).ravel()
        edges[s] = np.quantile(x, qs)
    return edges


def bucketize(feat: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """feat (N,2,4) -> buckets (N,2,4) int16"""
    b = np.zeros(feat.shape, dtype=np.int16)
    for s in range(4):
        b[:, :, s] = np.searchsorted(edges[s], feat[:, :, s].astype(np.float32), side="right")
    return b


class Flat:
    """Tree in flat arrays for fast traversal."""

    def __init__(self, tree: Tree):
        n = len(tree)
        self.n = n
        self.actor = np.array(tree.actor, dtype=np.int8)
        self.street = np.array(tree.street, dtype=np.int8)
        self.kind = np.array(tree.kind, dtype=np.int8)
        self.c0 = np.array([c[0] for c in tree.contrib], dtype=np.float32)
        self.c1 = np.array([c[1] for c in tree.contrib], dtype=np.float32)
        self.folder = np.array(tree.folder, dtype=np.int8)
        self.kids = [sorted(tree.children[i].items()) for i in range(n)]   # [(action, child)]


def worker(wid, args, flat_args, shm_names, counter, total_batches):
    feat_path, batch, seed = args
    data = np.load(feat_path)
    buckets_all = np.load(feat_path.replace(".npz", ".buckets.npy"), mmap_mode="r")
    out_all = data["out"]
    tree = Tree()
    F = Flat(tree)
    shmR = shared_memory.SharedMemory(name=shm_names[0])
    shmS = shared_memory.SharedMemory(name=shm_names[1])
    R = np.ndarray((F.n, NB, 4), dtype=np.float32, buffer=shmR.buf)
    S = np.ndarray((F.n, NB, 4), dtype=np.float32, buffer=shmS.buf)
    rng = np.random.default_rng(seed)
    N = len(out_all)

    while True:
        with counter.get_lock():
            t = counter.value
            if t >= total_batches:
                break
            counter.value += 1
        idx = rng.integers(0, N, size=batch)
        bk = np.asarray(buckets_all[idx], dtype=np.int64)        # (B,2,4)
        out = out_all[idx].astype(np.float32)
        w_avg = float(t + 1)                                      # linear averaging

        def walk(n, pi0, pi1):
            k = F.kind[n]
            if k == 1:     # fold
                return np.full(batch, -F.c0[n] if F.folder[n] == 0 else F.c1[n], dtype=np.float32)
            if k == 2:     # showdown (equal contributions)
                return (2.0 * out - 1.0) * F.c0[n]
            p = F.actor[n]
            b = bk[:, p, F.street[n]]
            kids = F.kids[n]
            acts = [a for a, _ in kids]
            reg = R[n][b][:, acts]                                # (B, nA)
            pos = np.maximum(reg, 0.0)
            ssum = pos.sum(axis=1, keepdims=True)
            strat = np.where(ssum > 0, pos / np.where(ssum > 0, ssum, 1.0), 1.0 / len(acts))
            vals = np.empty((batch, len(acts)), dtype=np.float32)
            for i, (a, c) in enumerate(kids):
                if p == 0:
                    vals[:, i] = walk(c, pi0 * strat[:, i], pi1)
                else:
                    vals[:, i] = walk(c, pi0, pi1 * strat[:, i])
            v = (strat * vals).sum(axis=1)
            if p == 0:
                inst = (vals - v[:, None]) * pi1[:, None]
                my_reach = pi0
            else:
                inst = -(vals - v[:, None]) * pi0[:, None]
                my_reach = pi1
            for i, a in enumerate(acts):
                R[n, :, a] += np.bincount(b, weights=inst[:, i], minlength=NB).astype(np.float32)
                S[n, :, a] += (w_avg * np.bincount(b, weights=my_reach * strat[:, i], minlength=NB)).astype(np.float32)
            np.maximum(R[n], 0.0, out=R[n])                      # CFR+
            return v

        walk(0, np.ones(batch, dtype=np.float32), np.ones(batch, dtype=np.float32))
        if wid == 0 and t % 10 == 0:
            print(f"  batch {t}/{total_batches}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deals")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    d = np.load(a.deals)
    feat = d["feat"]
    edges = make_edges(feat)
    np.save(a.deals.replace(".npz", ".buckets.npy"), bucketize(feat, edges))
    tree = Tree()
    n = len(tree)
    size = n * NB * 4 * 4
    shmR = shared_memory.SharedMemory(create=True, size=size)
    shmS = shared_memory.SharedMemory(create=True, size=size)
    R = np.ndarray((n, NB, 4), dtype=np.float32, buffer=shmR.buf)
    S = np.ndarray((n, NB, 4), dtype=np.float32, buffer=shmS.buf)
    R[:] = 0
    S[:] = 0
    counter = Value("i", 0)
    t = time.time()
    procs = [Process(target=worker, args=(w, (a.deals, a.batch, 77 + w), None,
                                          (shmR.name, shmS.name), counter, a.batches))
             for w in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"trained {a.batches} x {a.batch} deals in {time.time() - t:.0f}s")

    strat = np.zeros((n, NB, 4), dtype=np.float32)
    for i in range(n):
        if tree.kind[i] != 0:
            continue
        acts = sorted(tree.children[i].keys())
        s = S[i][:, acts]
        tot = s.sum(axis=1, keepdims=True)
        s = np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / len(acts))
        strat[i][:, acts] = s
    np.savez_compressed(a.out, strat=strat.astype(np.float16), edges=edges)
    print("saved", a.out, os.path.getsize(a.out) // 1024, "KB")
    shmR.close(); shmR.unlink(); shmS.close(); shmS.unlink()


if __name__ == "__main__":
    main()
