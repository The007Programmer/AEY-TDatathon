"""House Edge forecaster: Bayesian lag-pair XNOR win model with change points + lag-dependent payout-size model.
numpy only. Deterministic (no randomness)."""
import csv, itertools, math
from collections import defaultdict
import numpy as np

PAYOUTS = [0, 1, 2, 5, 20]
PAIRS = list(itertools.combinations(range(1, 9), 2))
COMMON = False

# ---------- log-gamma (Lanczos, g=7, n=9), vectorized ----------
_LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
       -176.61502916214059, 12.507343278686905, -0.13857109526572012,
       9.9843695780195716e-6, 1.5056327351493116e-7]
def lgamma(x):
    x = np.asarray(x, dtype=float)
    # all our arguments are >= 0.3 > 0.5? handle x<0.5 by reflection
    small = x < 0.5
    xx = np.where(small, 1.0 - x, x) - 1.0
    a = np.full_like(xx, _LG[0])
    t = xx + 7.5
    for i in range(1, 9):
        a = a + _LG[i] / (xx + i)
    res = 0.5 * math.log(2 * math.pi) + (xx + 0.5) * np.log(t) - t + np.log(a)
    if small.any():
        res = np.where(small, math.log(math.pi) - np.log(np.abs(np.sin(math.pi * x))) - res, res)
    return res
def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)
def bb(k1, k0, a, b):
    return betaln(a + k1, b + k0) - betaln(a, b)

# ---------- IO ----------
def load(split_dir):
    queries = []
    with open(f"{split_dir}/queries.csv", newline="") as f:
        for r in csv.DictReader(f):
            queries.append(dict(query_id=r["query_id"], episode_id=r["episode_id"], machine_id=r["machine_id"],
                                machine_type=r["machine_type"], signal=int(r["signal"]), cutoff=int(r["cutoff"])))
    H = defaultdict(dict)
    with open(f"{split_dir}/history.csv", newline="") as f:
        for r in csv.DictReader(f):
            H[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    return queries, H

def write(path, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid, p in preds:
            w.writerow([qid] + [repr(float(x)) for x in p])

# ---------- win model ----------
DEFAULT = dict(a0=0.5, S=2.5, step=4, pchange=0.5, tmin=40, other=1e-4, p4=0.1, lsh=0.0, tps=25.0,
               mainp={(1, 3): 1.0, (2, 5): 1.0, (4, 7): 1.0, (1, 2): 0.1},
               levels={"steady": (0.95, 0.15), "balanced": (0.86, 0.06), "jackpot": (0.52, 0.012)})

def _arrays(s, cutoff, pr):
    T = []; A = []; B = []; Y = []
    for t in range(8, cutoff):
        v = s.get(t)
        if v is None: continue
        if COMMON and any(s.get(t - l) is None for l in range(1, 9)): continue   # same tick set for every pair
        a = s.get(t - pr[0]); b = s.get(t - pr[1])
        if a is None or b is None: continue
        T.append(t); A.append(a > 0); B.append(b > 0); Y.append(v > 0)
    return np.array(T, int), np.array(A, int), np.array(B, int), np.array(Y, int)

def _evid(c, cnt_pre, cnt_all):
    a = c["a0"]
    k1p = cnt_pre[..., 1]; k0p = cnt_pre[..., 0]
    llpre = bb(k1p, k0p, a, a).sum(-1)
    mpre = (k1p + a) / (k1p + k0p + 2 * a)
    A = c["S"] * (1 - mpre) + 0.3; B = c["S"] * mpre + 0.3
    k1q = cnt_all[None, :, 1] - k1p; k0q = cnt_all[None, :, 0] - k0p
    return llpre + bb(k1q, k0q, A, B).sum(-1), (k1q + A) / (k1q + k0q + A + B)

def _evid_x(c, cnt_pre, cnt_all, mtype):
    """2-cell XNOR model (cell0=lags differ, cell1=lags agree) with a type-level prior mixing both directions."""
    hi, lo = c["levels"][mtype]; n = c["tps"]
    k1p = cnt_pre[..., 1]; k0p = cnt_pre[..., 0]
    comps = []
    for d in (0, 1):
        m = np.array([lo, hi]) if d == 1 else np.array([hi, lo])
        A0 = n * m; B0 = n * (1 - m)
        comps.append((bb(k1p, k0p, A0, B0).sum(-1), (k1p + A0) / (k1p + k0p + A0 + B0)))
    Lmix = np.logaddexp(comps[0][0], comps[1][0])
    L = Lmix - math.log(2)
    w1 = np.exp(comps[1][0] - Lmix)
    mpre = w1[..., None] * comps[1][1] + (1 - w1[..., None]) * comps[0][1]
    A = c["S"] * (1 - mpre) + 0.3; B = c["S"] * mpre + 0.3
    k1q = cnt_all[None, :, 1] - k1p; k0q = cnt_all[None, :, 0] - k0p
    return L + bb(k1q, k0q, A, B).sum(-1), (k1q + A) / (k1q + k0q + A + B), L, mpre

def _machine(c, hist, cutoff, grid, mtype):
    out = []
    for pr in PAIRS:
        prior = c["mainp"].get(pr, c["other"])
        T, A, B, Y = _arrays(hist, cutoff, pr)
        if len(T) < 6: continue
        a = hist.get(cutoff - pr[0]); b = hist.get(cutoff - pr[1])
        for model in ("x", "f"):
            if model == "x": C = (A == B).astype(int); nc = 2; mp = 1 - c["p4"]
            else: C = 2 * A + B; nc = 4; mp = c["p4"]
            cnt = np.zeros((len(grid) + 1, nc, 2))
            for g, tau in enumerate(list(grid) + [cutoff + 1]):
                m = T < tau; np.add.at(cnt[g], (C[m], Y[m]), 1)
            if model == "x" and c["tps"] > 0:
                lla, pta, Lpre, mpre = _evid_x(c, cnt, cnt[-1], mtype)   # last row = no change
                llt, pt = lla[:-1], pta[:-1]; lln = float(Lpre[-1]); pn = mpre[-1]
            else:
                llt, pt = _evid(c, cnt[:-1], cnt[-1])
                ca = cnt[-1]; lln = float(bb(ca[:, 1], ca[:, 0], c["a0"], c["a0"]).sum())
                pn = (ca[:, 1] + c["a0"]) / (ca.sum(1) + 2 * c["a0"])
            if a is None or b is None:
                predn = pn.mean(); predt = pt.mean(1)
            else:
                cell = int(int(a > 0) == int(b > 0)) if model == "x" else 2 * int(a > 0) + int(b > 0)
                predn = pn[cell]; predt = pt[:, cell]
            out.append((math.log(prior * mp), lln, llt, predn, predt))
    return out

def predict_win_episode(c, rows, H):
    cutoff = rows[0]["cutoff"]
    grid = np.arange(c["tmin"], cutoff - 1, c["step"]); G = len(grid)
    mach = []
    for r in rows:
        hy = _machine(c, H[r["machine_id"]], cutoff, grid, r["machine_type"])
        lp = np.array([h[0] for h in hy]); LN = np.array([h[1] for h in hy]); LT = np.array([h[2] for h in hy])
        PN = np.array([h[3] for h in hy]); PT = np.array([h[4] for h in hy])
        en = np.logaddexp.reduce(lp + LN); et = np.logaddexp.reduce(lp[:, None] + LT, axis=0)
        mach.append((lp, LN, LT, PN, PT, en, et))
    EN = sum(m[5] for m in mach); ET = sum(m[6] for m in mach)
    lpn = math.log(1 - c["pchange"]); lpt = math.log(c["pchange"] / G)
    ev_sh = np.logaddexp(EN + lpn, np.logaddexp.reduce(ET + lpt))
    ev_in = sum(np.logaddexp(m[5] + lpn, np.logaddexp.reduce(m[6] + lpt)) for m in mach)
    wsh = 1 / (1 + math.exp(float(np.clip(ev_in - ev_sh - c["lsh"], -50, 50))))
    out = {}
    for r, (lp, LN, LT, PN, PT, en, et) in zip(rows, mach):
        preds = []
        for mode in ("sh", "in"):
            if mode == "sh": on = (EN - en) + lpn; ot = (ET - et) + lpt
            else: on = lpn; ot = np.full(G, lpt)
            Wn = lp + LN + on; Wt = lp[:, None] + LT + ot[None, :]
            m = max(Wn.max(), Wt.max()); wn = np.exp(Wn - m); wt = np.exp(Wt - m); Z = wn.sum() + wt.sum()
            preds.append(float((wn @ PN + (wt * PT).sum()) / Z))
        out[r["query_id"]] = wsh * preds[0] + (1 - wsh) * preds[1]
    return out

# ---------- payout-size model ----------
# log(p5/p2), log(p20/p5) fixed per machine type (estimated from validation targets; std 0 across machines)
FAM = {"steady": (-0.705, -1.406, -3.892), "balanced": (0.210, -0.981, -1.792), "jackpot": (0.265, 0.251, -1.099)}
_MU = np.linspace(-3, 3, 61); _BE = np.linspace(-4, 4, 81)
_M, _B = np.meshgrid(_MU, _BE, indexing="ij")
SIZE_DEFAULT = dict(sd_mu=0.6, sd_b=1.2, fam=FAM, lag_prior={1: 1.0, 2: 1.0, 4: 1.0, 3: 0.15, 5: 0.15, 6: 0.15, 7: 0.15, 8: 0.15})
def size_lag(hist, cutoff, mtype, sd_mu=0.6, sd_b=1.2, fam=FAM, lag_prior=None, b_mode=0.0):
    lag_prior = lag_prior or SIZE_DEFAULT["lag_prior"]
    lags = sorted(lag_prior)
    mu0, r52, r205 = fam[mtype]
    w = np.array([1.0, math.exp(r52), math.exp(r52 + r205)]); big = w.sum()
    if b_mode > 0:
        lp0 = -0.5 * ((_M - mu0) / sd_mu) ** 2 + np.logaddexp(-0.5 * ((_B - b_mode) / sd_b) ** 2, -0.5 * ((_B + b_mode) / sd_b) ** 2)
    else:
        lp0 = -0.5 * ((_M - mu0) / sd_mu) ** 2 - 0.5 * (_B / sd_b) ** 2
    lp0 = lp0 - np.logaddexp.reduce(lp0.ravel())
    wins = [(t, v) for t, v in hist.items() if t < cutoff and v > 0]
    past = [v for t, v in hist.items() if t < cutoff]
    bigrate = (sum(v >= 2 for v in past) + 1) / (len(past) + 2)
    p0 = 1 / (1 + np.exp(_M) * big); p1 = 1 / (1 + np.exp(_M + _B) * big)
    lq0, lr0, lq1, lr1 = np.log(p0), np.log1p(-p0), np.log(p1), np.log1p(-p1)
    pu = bigrate * p1 + (1 - bigrate) * p0          # likelihood for wins whose lag-l payout is unobserved
    lqu, lru = np.log(pu), np.log1p(-pu)
    evid = []; preds = []
    for l in lags:
        n = np.zeros((3, 2))  # rows: lag small, lag big, lag unknown
        for t, v in wins:
            u = hist.get(t - l)
            n[2 if u is None else int(u >= 2), int(v == 1)] += 1.0
        L = lp0 + n[0, 1] * lq0 + n[0, 0] * lr0 + n[1, 1] * lq1 + n[1, 0] * lr1 + n[2, 1] * lqu + n[2, 0] * lru
        Z = np.logaddexp.reduce(L.ravel()); post = np.exp(L - Z)
        u = hist.get(cutoff - l)
        if u is None: P1 = (post * pu).sum()
        else: P1 = (post * (p1 if u >= 2 else p0)).sum()
        evid.append(Z + math.log(lag_prior[l])); preds.append(P1)
    evid = np.array(evid); ww = np.exp(evid - evid.max()); ww /= ww.sum()
    P1 = float(ww @ np.array(preds))
    return np.array([P1, *((1 - P1) * w / big)])

def size_dirichlet(hist, cutoff, mtype, prior=None, strength=10.0):
    """Fallback if the one-parameter size family does not hold: type-mean Dirichlet prior + machine win counts."""
    base = np.array(prior[mtype] if prior else [0.4, 0.35, 0.2, 0.05])
    c = base * strength
    for t, v in hist.items():
        if t < cutoff and v > 0: c[PAYOUTS.index(v) - 1] += 1
    return c / c.sum()

def _episode_job(args):
    rows, Hs, c, size_cfg = args
    pw = predict_win_episode(c, rows, Hs)
    out = {}
    for r in rows:
        if size_cfg.get("mode", "family") == "dirichlet":
            sd = size_dirichlet(Hs[r["machine_id"]], r["cutoff"], r["machine_type"], size_cfg.get("dir_prior"))
        else:
            sd = size_lag(Hs[r["machine_id"]], r["cutoff"], r["machine_type"],
                          **{k: v for k, v in size_cfg.items() if k in ("sd_mu", "sd_b", "fam", "lag_prior", "b_mode")})
        p = np.concatenate([[1 - pw[r["query_id"]]], pw[r["query_id"]] * sd])
        p = np.clip(p, 1e-7, 1); p = p / p.sum()
        out[r["query_id"]] = p
    return out

def predict_all(queries, H, cfg=None, size_cfg=None, workers=1):
    """Episodes are independent, so results are identical for any number of workers."""
    c = dict(DEFAULT); c.update(cfg or {})
    size_cfg = dict(SIZE_DEFAULT, **(size_cfg or {}))
    eps = defaultdict(list)
    for q in queries: eps[q["episode_id"]].append(q)
    jobs = [(eps[ep], {r["machine_id"]: H[r["machine_id"]] for r in eps[ep]}, c, size_cfg) for ep in sorted(eps)]
    res = {}
    if workers > 1:
        try:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                for o in pool.map(_episode_job, jobs, chunksize=4): res.update(o)
        except Exception:
            res = {}
    if not res:
        for j in jobs: res.update(_episode_job(j))
    return [(q["query_id"], res[q["query_id"]]) for q in queries]
