"""
Score one or more bot specs against the whole sparring field at once.

    .venv/bin/python -m train.suite agent.py "agent.py?rf_oop=0.8" --seeds 8
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from multiprocessing import Pool

from train.arena import play_match

FIELD = [
    "baseline-bots/baseline_random.py",
    "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_half.py",
    "baseline-bots/baseline_always_pot.py",
    "train/opps/equity_simple.py",
    "train/opps/tag.py",
    "train/opps/maniac.py",
    "train/opps/nit.py",
    "train/opps/slow_mc.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--field", nargs="*", default=None)
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args()
    field = a.field or [f for f in FIELD if not (a.skip_slow and "slow_mc" in f)]
    jobs = [(s, o, a.seed0 + i, a.deals) for s in a.specs for o in field for i in range(a.seeds)]
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = pool.map(play_match, jobs, chunksize=1)
    by = {}
    for (s, o, _, _), r in zip(jobs, res):
        by.setdefault(s, {}).setdefault(o, []).append(r)
    names = [os.path.basename(o).replace("baseline_", "b_").replace(".py", "") for o in field]
    print(f"{len(jobs)} matches in {time.time() - t:.0f}s")
    print(f"{'spec':34s}" + "".join(f"{n[:9]:>10s}" for n in names) + "   points   errs  maxT")
    for s in a.specs:
        row, pts, errs, maxt = [], 0.0, 0, 0.0
        for o in field:
            rs = by[s][o]
            row.append(statistics.mean(r["mean"] for r in rs))
            pts += sum(1.0 if r["verdict"] == "WIN" else 0.5 if r["verdict"] == "CHOP" else 0.0 for r in rs)
            for r in rs:
                for h in r["halves"]:
                    errs += h["a1"]["errors"]
                    maxt = max(maxt, h["a1"]["time"])
        total = len(field) * a.seeds
        print(f"{s[:34]:34s}" + "".join(f"{v:+10.2f}" for v in row)
              + f"   {pts:5.1f}/{total}  {errs:4d}  {maxt:.2f}s")


if __name__ == "__main__":
    main()
