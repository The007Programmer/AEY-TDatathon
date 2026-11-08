"""Estimate round-specific priors from the round's public data (all histories + validation targets).
Writes config.json used by predict.py. Deterministic."""
import argparse, itertools, json, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hemodel import load, bb, PAIRS, _M, _B

def win_pair_stats(splits_data):
    """Per machine, log-evidence of the 2-cell XNOR model for every lag pair (no change point; uses
    second half of history too, flips just lower evidence a bit). Returns list of (type, ev[pairs], rates)."""
    out = []
    for q, H in splits_data:
        for r in q:
            s = H[r["machine_id"]]; W = {t: int(v > 0) for t, v in s.items() if t < r["cutoff"]}
            ev = []; rates = []
            for pr in PAIRS:
                k = np.zeros((2, 2))
                for t, y in W.items():
                    a = W.get(t - pr[0]); b = W.get(t - pr[1])
                    if a is None or b is None: continue
                    k[int(a == b), y] += 1
                ev.append(float(bb(k[:, 1], k[:, 0], 0.5, 0.5).sum()))
                rates.append(((k[:, 1] + 0.5) / (k.sum(1) + 1)).tolist() + [k.sum()])
            out.append((r["machine_type"], np.array(ev), rates))
    return out

def em_prior(evs, iters=200, floor=1e-4):
    E = np.array(evs); K = E.shape[1]; pi = np.full(K, 1.0 / K)
    for _ in range(iters):
        L = E + np.log(pi); L -= L.max(1, keepdims=True); R = np.exp(L); R /= R.sum(1, keepdims=True)
        pi = R.mean(0); pi = np.maximum(pi, floor); pi /= pi.sum()
    return pi, R

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    splits = [load(os.path.join(a.data, sp)) for sp in ("train", "validation", "test")]
    stats = win_pair_stats(splits)
    pi, R = em_prior([s[1] for s in stats])
    order = np.argsort(-pi)
    top = pi[order[0]]
    # p4: share of machines better explained by an unrestricted 4-cell table than by XNOR (on their MAP pair)
    ev2 = []
    for q, H in splits:
        for r in q:
            s_ = H[r["machine_id"]]; W = {t: int(v > 0) for t, v in s_.items() if t < r["cutoff"]}
            best = None
            for i in order[:3]:
                pr = PAIRS[i]; k2 = np.zeros((2, 2)); k4 = np.zeros((4, 2))
                for t, y in W.items():
                    a_ = W.get(t - pr[0]); b_ = W.get(t - pr[1])
                    if a_ is None or b_ is None: continue
                    k2[int(a_ == b_), y] += 1; k4[2 * a_ + b_, y] += 1
                e2 = float(bb(k2[:, 1], k2[:, 0], 0.5, 0.5).sum()); e4 = float(bb(k4[:, 1], k4[:, 0], 0.5, 0.5).sum())
                if best is None or max(e2, e4) > max(best): best = (e2, e4)
            ev2.append(list(best))
    p4pi, _ = em_prior(ev2, floor=0.02)
    p4 = float(p4pi[1]); print("XNOR vs 4-cell share:", np.round(p4pi, 3))
    # keep only clearly recurring pairs (EM mass >= 5%); others get a tiny prior (flipped machines leak mass to spurious pairs)
    mainp = {f"{PAIRS[i][0]},{PAIRS[i][1]}": (float(pi[i] / top) if pi[i] >= 0.05 else 1e-4) for i in range(len(PAIRS))}
    print("top pairs:", [(PAIRS[i], round(float(pi[i]), 3)) for i in order[:6]])
    import csv
    qv = {r["query_id"]: r for r in splits[1][0]}
    tgt = defaultdict(list)
    with open(os.path.join(a.data, "validation", "targets.csv")) as f:
        for r in csv.DictReader(f):
            tgt[qv[r["query_id"]]["machine_type"]].append([float(r[k]) for k in ("p_0", "p_1", "p_2", "p_5", "p_20")])
    # win levels: 2-means on logit(win) of validation targets, per type
    levels = {}
    for ty, v in tgt.items():
        w = 1 - np.array(v)[:, 0]; z = np.log(w / (1 - w)); cc = np.array([z.min(), z.max()])
        for _ in range(100):
            asg = np.abs(z[:, None] - cc[None]).argmin(1); cc = np.array([z[asg == k].mean() for k in (0, 1)])
        levels[ty] = [float(np.median(w[asg == 1])), float(np.median(w[asg == 0]))]
    print("levels:", levels)
    # size family: log(p2/p1) mean, log(p5/p2), log(p20/p5) medians
    fam = {}; fam_ok = True; dir_prior = {}
    for ty, v in tgt.items():
        vv = np.array(v); wsh = vv[:, 1:] / vv[:, 1:].sum(1, keepdims=True); dir_prior[ty] = wsh.mean(0).tolist()
        v = np.array(v)
        r = np.c_[np.log(v[:, 2] / v[:, 1]), np.log(v[:, 3] / v[:, 2]), np.log(v[:, 4] / v[:, 3])]
        fam[ty] = [float(r[:, 0].mean()), float(np.median(r[:, 1])), float(np.median(r[:, 2]))]
        print(ty, "size ratios mean", np.round(r.mean(0), 3), "sd", np.round(r.std(0), 3))
        if r[:, 1].std() > 0.05 or r[:, 2].std() > 0.05: fam_ok = False
    if not fam_ok: print("WARNING: size family ratios vary -> one-parameter size model invalid; using Dirichlet fallback")
    # size lag prior: EM over lags 1..8 using the size model's marginal likelihood
    sd_mu, sd_b = 0.6, 1.2
    E = []; size_recs = []
    for q, H in splits:
        for r in q:
            mu0, r52, r205 = fam[r["machine_type"]]; big = 1 + math.exp(r52) + math.exp(r52 + r205)
            lp0 = -0.5 * ((_M - mu0) / sd_mu) ** 2 - 0.5 * (_B / sd_b) ** 2; lp0 -= np.logaddexp.reduce(lp0.ravel())
            p0 = 1 / (1 + np.exp(_M) * big); p1 = 1 / (1 + np.exp(_M + _B) * big)
            s_ = H[r["machine_id"]]; ev = []
            n_all = np.zeros(2)
            for t, v in s_.items():
                if t < r["cutoff"] and v > 0: n_all[int(v == 1)] += 1
            lm = -0.5 * ((_M[:, 0] - mu0) / sd_mu) ** 2; lm -= np.logaddexp.reduce(lm); pm = 1 / (1 + np.exp(_M[:, 0]) * big)
            ev_null = float(np.logaddexp.reduce(lm + n_all[1] * np.log(pm) + n_all[0] * np.log1p(-pm)))
            past = [v for t, v in s_.items() if t < r["cutoff"]]
            q_ = (sum(v >= 2 for v in past) + 1) / (len(past) + 2); pu = q_ * p1 + (1 - q_) * p0
            n3s = []
            for l in range(1, 9):
                n = np.zeros((3, 2))
                for t, v in s_.items():
                    if t >= r["cutoff"] or v <= 0: continue
                    u = s_.get(t - l)
                    n[2 if u is None else int(u >= 2), int(v == 1)] += 1
                n3s.append(n)
                L = (lp0 + n[0, 1] * np.log(p0) + n[0, 0] * np.log1p(-p0) + n[1, 1] * np.log(p1) + n[1, 0] * np.log1p(-p1)
                     + n[2, 1] * np.log(pu) + n[2, 0] * np.log1p(-pu))
                ev.append(float(np.logaddexp.reduce(L.ravel())))
            E.append([ev_null] + ev); size_recs.append((r["machine_type"], n3s, q_))
    lpi, _ = em_prior(E, floor=0.005)
    print("size EM (null, lags 1..8):", np.round(lpi, 3))
    lw = lpi[1:]
    lag_prior = {str(l): float(max(lw[l - 1] / lw.max(), 0.02)) for l in range(1, 9)}
    print("size lag prior:", {k: round(v, 3) for k, v in lag_prior.items()})
    # empirical-Bayes for the size effect prior (bimodal +/-b_mode vs gaussian), on all histories
    best = None
    lagw = np.array([lag_prior[str(l)] for l in range(1, 9)]); lagw = lagw / lagw.sum()
    for sdm, bm, sdb in [(0.6, 0.0, 1.2), (0.6, 0.8, 0.1), (0.6, 1.0, 0.1), (0.6, 1.2, 0.1), (0.6, 1.4, 0.1), (0.6, 1.7, 0.1), (0.6, 1.2, 0.3), (0.45, 1.2, 0.1), (0.8, 1.2, 0.1)]:
        tot = 0.0
        for rec in size_recs:
            ty, n3s, q_ = rec
            mu0, r52, r205 = fam[ty]; big = 1 + math.exp(r52) + math.exp(r52 + r205)
            if bm > 0: lp0 = -0.5 * ((_M - mu0) / sdm) ** 2 + np.logaddexp(-0.5 * ((_B - bm) / sdb) ** 2, -0.5 * ((_B + bm) / sdb) ** 2)
            else: lp0 = -0.5 * ((_M - mu0) / sdm) ** 2 - 0.5 * (_B / sdb) ** 2
            lp0 = lp0 - np.logaddexp.reduce(lp0.ravel())
            p0 = 1 / (1 + np.exp(_M) * big); p1 = 1 / (1 + np.exp(_M + _B) * big); pu = q_ * p1 + (1 - q_) * p0
            ev = []
            for l, n in enumerate(n3s):
                if lagw[l] < 0.05: continue
                L = (lp0 + n[0, 1] * np.log(p0) + n[0, 0] * np.log1p(-p0) + n[1, 1] * np.log(p1) + n[1, 0] * np.log1p(-p1)
                     + n[2, 1] * np.log(pu) + n[2, 0] * np.log1p(-pu))
                ev.append(float(np.logaddexp.reduce(L.ravel())) + math.log(lagw[l]))
            tot += float(np.logaddexp.reduce(ev))
        print("size EB", (sdm, bm, sdb), round(tot, 1))
        if best is None or tot > best[0]: best = (tot, sdm, bm, sdb)
    _, sdm, bm, sdb = best
    cfg = dict(win=dict(mainp=mainp, levels=levels, p4=min(max(p4, 0.02), 0.5)),
               size=dict(fam=fam, lag_prior=lag_prior, sd_mu=sdm, b_mode=bm, sd_b=sdb,
                         mode="family" if fam_ok else "dirichlet", dir_prior=dir_prior))
    print("CONFIG:", json.dumps(cfg))
    json.dump(cfg, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
