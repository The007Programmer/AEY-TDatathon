"""
In-process duplicate-match arena (no HTTP), parallel over CPU cores.

Plays exactly like the judge: same deals (game.make_deal), seats swapped in
half 2, fresh memory every half, errors -> default action, 5 errors =
forfeit, time bank per half, paired bb/deal + 95% CI verdict.

Usage:
    .venv/bin/python -m train.arena agent.py baseline-bots/baseline_random.py --seeds 8 --deals 1000
    .venv/bin/python -m train.arena "agent.py?rf_oop=0.8" agent.py --seeds 16

A "?k=v&k2=v2" suffix sets CFG_OVERRIDE on that bot (our agent.py only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import (ERROR_LIMIT, TIME_BANK_SECONDS, NETWORK_GRACE_SECONDS, Action,  # noqa: E402
                  AgentConfig, Game, button_for_deal, default_action, make_deal)

_LOADED = {}


def load_bot(spec: str):
    if spec in _LOADED:
        return _LOADED[spec]
    path, _, query = spec.partition("?")
    overrides = {}
    for kv in filter(None, query.split("&")):
        k, v = kv.split("=")
        try:
            overrides[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            overrides[k] = v
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.dirname(full) not in sys.path:
        sys.path.insert(0, os.path.dirname(full))
    name = f"bot_{len(_LOADED)}_{abs(hash(spec))}"
    mod_spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    if overrides:
        mod.CFG_OVERRIDE = overrides
    _LOADED[spec] = mod
    return mod


class Seat:
    def __init__(self, mod):
        self.mod = mod
        self.memory = {}
        self.time = 0.0
        self.errors = 0
        self.forfeit = False
        self.calls = 0
        self.max_call = 0.0
        self.first_error = None

    def _timed(self, fn, *args):
        t = time.perf_counter()
        try:
            out = fn(*args)
            ok = True
        except Exception:
            out, ok = None, False
            if self.first_error is None:
                self.first_error = traceback.format_exc()
        dt = time.perf_counter() - t
        self.time += max(0.0, dt - NETWORK_GRACE_SECONDS)
        self.max_call = max(self.max_call, dt)
        if dt > 1.0:
            ok = False
        return out, ok

    def start(self, cfg):
        _, ok = self._timed(self.mod.new_half, cfg, self.memory)
        if not ok:
            self._err()

    def _err(self):
        self.errors += 1
        if self.errors >= ERROR_LIMIT:
            self.forfeit = True

    def act(self, obs):
        if self.forfeit or self.time >= TIME_BANK_SECONDS:
            return default_action(obs.to_call)
        self.calls += 1
        a, ok = self._timed(self.mod.choose_action, obs, self.memory)
        if not ok or not isinstance(a, Action) or a not in obs.legal_actions:
            self._err()
            return default_action(obs.to_call)
        return a

    def hand_end(self, hr):
        if self.forfeit or self.time >= TIME_BANK_SECONDS:
            return
        _, ok = self._timed(self.mod.on_hand_end, hr, self.memory)
        if not ok:
            self._err()


def play_half(spec1, spec2, seed, half, deals):
    m1, m2 = load_bot(spec1), load_bot(spec2)
    s1, s2 = Seat(m1), Seat(m2)
    seats = (s1, s2) if half == 1 else (s2, s1)
    cfg = AgentConfig(num_hands=deals)
    for s in seats:
        s.start(cfg)
    nets1 = []
    for d in range(deals):
        g = Game(make_deal(str(seed), d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            seat = g.to_act
            g.apply(seats[seat].act(g.observation(seat)))
        for seat in (0, 1):
            seats[seat].hand_end(g.hand_result(seat))
        a1_seat = 0 if half == 1 else 1
        nets1.append(g.result().nets[a1_seat])
    stats = {n: {"time": round(s.time, 3), "errors": s.errors + s.memory.get("_fallbacks", 0), "forfeit": s.forfeit,
                 "calls": s.calls, "max_call_ms": round(s.max_call * 1000, 1),
                 "first_error": s.first_error}
             for n, s in (("a1", s1), ("a2", s2))}
    return nets1, stats


def play_match(args):
    spec1, spec2, seed, deals = args
    n1, st1 = play_half(spec1, spec2, seed, 1, deals)
    n2, st2 = play_half(spec1, spec2, seed, 2, deals)
    paired = [(a + b) / 2.0 for a, b in zip(n1, n2)]       # chips / 2 = bb
    mean = sum(paired) / len(paired)
    sd = statistics.stdev(paired)
    se = sd / math.sqrt(len(paired))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    forfeit1 = st1["a1"]["forfeit"] or st2["a1"]["forfeit"]
    forfeit2 = st1["a2"]["forfeit"] or st2["a2"]["forfeit"]
    if forfeit1 and forfeit2:
        v = "BOTH_FORFEIT"
    elif forfeit1:
        v = "LOSS"
    elif forfeit2:
        v = "WIN"
    else:
        v = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    return {"seed": seed, "mean": mean, "sd": sd, "lo": lo, "hi": hi, "verdict": v,
            "halves": [st1, st2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a1")
    ap.add_argument("a2")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--json")
    a = ap.parse_args()
    jobs = [(a.a1, a.a2, a.seed0 + i, a.deals) for i in range(a.seeds)]
    t = time.time()
    with Pool(min(a.procs, len(jobs))) as pool:
        res = pool.map(play_match, jobs, chunksize=1)
    print(f"{a.a1}  vs  {a.a2}   ({a.seeds} matches x {a.deals} deals, {time.time() - t:.0f}s)")
    for r in res:
        h = r["halves"]
        t1 = max(h[0]["a1"]["time"], h[1]["a1"]["time"])
        t2 = max(h[0]["a2"]["time"], h[1]["a2"]["time"])
        e1 = h[0]["a1"]["errors"] + h[1]["a1"]["errors"]
        e2 = h[0]["a2"]["errors"] + h[1]["a2"]["errors"]
        mx = max(h[0]["a1"]["max_call_ms"], h[1]["a1"]["max_call_ms"])
        print(f"  seed {r['seed']}: {r['mean']:+7.3f} bb/deal  CI [{r['lo']:+.3f},{r['hi']:+.3f}]  "
              f"{r['verdict']:5s}  time a1 {t1:.2f}s (max call {mx:.0f}ms) a2 {t2:.2f}s  errors {e1}/{e2}")
        for half in h:
            if half["a1"]["first_error"]:
                print(half["a1"]["first_error"])
                break
    means = [r["mean"] for r in res]
    w = sum(r["verdict"] == "WIN" for r in res)
    l = sum(r["verdict"] == "LOSS" for r in res)
    c = len(res) - w - l
    print(f"  AVG {statistics.mean(means):+.3f} bb/deal   W/C/L = {w}/{c}/{l}   "
          f"points {w + 0.5 * c:.1f}/{len(res)}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
