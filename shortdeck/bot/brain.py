"""
Decision engine: equity vs a modelled range + one-step expected-value search.

For every legal action we estimate the chips we end up with from here on
(relative to folding now = 0):

    check/call : E * (pot after call) - call
    raise      : f * pot                                  (they fold)
               + (1 - f - r) * (E_cont * pot_called - a)  (they call)
               + r * max(-a, E_rr * pot_3bet - a - x)     (they re-raise)

f and r come from the opponent model (how often THIS opponent folds/raises
when facing a bet), and E_cont / E_rr are our equity against the stronger
part of their range that continues. Against calling stations f ~ 0, so we
only value-bet; against folders f is high, so we bluff a lot; against a bot
whose time bank ran out we raise every hand.
"""

from __future__ import annotations

import time

import numpy as np

from game import Action, Street

from . import equity as EQ
from . import opponent as OM

try:
    from . import blueprint as _BP
except Exception:          # blueprint is optional
    _BP = None

TIME_BANK = 8.0
K_MIN = 96
DEFAULTS = {
    "safety": 0.25,      # our own compute budget: 25% of the bank (network overhead is charged too)
    "k_max": 500,
    "tau": 0.08,         # softness of range thresholds
    "rf_oop": 0.88,      # equity realisation before the river when out of position
    "rf_ip": 1.0,
    "rf_mode": 0,
    "bp_mode": 8,        # bits: 1 blueprint preflop, 2 blueprint first bp_hands hands,
                         #       4 hybrid: blueprint unless exploiting gains > bp_margin * pot
    "bp_hands": 150,
    # bit 8: auto - blueprint while the opponent looks balanced, hybrid/exploit once it doesn't
    "bp_margin": 0.05,
    "bp_min_reach": 0.0,  # infosets the blueprint rarely trained -> exploit engine
}
TAU = DEFAULTS["tau"]
BOARD_N = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


def _real(e: float, rf: float, mode: int) -> float:
    """Equity we actually realise. mode 1: weak hands realise less (they get
    bet off their equity), strong hands close to all of it."""
    if rf >= 1.0:
        return e
    if mode == 1:
        return e * (rf + (1.0 - rf) * e)
    return e * rf


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def new_half(memory: dict, overrides: dict | None = None) -> None:
    memory.clear()
    memory["cfg"] = {**DEFAULTS, **(overrides or {})}
    memory["model"] = OM.new_model()
    memory["rng"] = np.random.default_rng()
    memory["spent"] = 0.0
    memory["sec_per_sample"] = 2.5e-6
    memory["decisions"] = 0
    memory["cur_hand"] = -1
    memory["n_seen"] = 0
    memory["cur_events"] = []
    memory["my_last"] = None
    memory["num_hands"] = 1000


# ---------------------------------------------------------------------------

def _strength(sim, street_n: int) -> np.ndarray:
    """Opponent strength per sample at a street (0 = preflop)."""
    if street_n == 0:
        return sim.pf_pct
    s = sim.street_pct[street_n]
    if street_n in sim.draw:
        s = np.where(sim.draw[street_n], np.maximum(s, 0.62 if street_n == 3 else 0.55), s)
    return s


def _range_weights(events, sim, model, TAU=TAU) -> np.ndarray:
    w = np.ones(sim.k)
    for street, is_me, key, kind in events:
        if is_me:
            continue
        g = key[0]
        lam = OM.reliability(model, g)
        s = _strength(sim, BOARD_N[Street(street)])
        if kind == "raise":
            thr = 1.0 - OM.freq(model, key, "raise")
            f = _sig((s - thr) / TAU)
        elif kind == "call":
            cont = OM.freq(model, key, "call") + OM.freq(model, key, "raise")
            thr = 1.0 - cont
            f = _sig((s - thr) / TAU)
            if OM.freq(model, key, "raise") > 0.02:
                thr_r = 1.0 - OM.freq(model, key, "raise")
                f = f * (1.0 - 0.35 * _sig((s - thr_r) / TAU))     # strong hands usually raise
        elif kind == "check":
            thr_b = 1.0 - OM.freq(model, key, "raise")
            f = 1.0 - 0.6 * _sig((s - thr_b) / TAU)                 # checks cap the range
        else:
            continue
        w *= (1.0 - lam) + lam * f
    return w


def _choose_k(obs, memory) -> int:
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    dec = memory["decisions"]
    per_hand = (dec / (obs.hand_index + 1)) if obs.hand_index > 20 else 3.0
    cfg = memory["cfg"]
    budget = TIME_BANK * cfg["safety"] - memory["spent"]
    target = budget / (hands_left * max(per_hand, 1.0))
    # spend more on big pots, less on tiny ones
    target *= 0.6 + min(1.6, obs.pot / 24.0)
    k = int(target / memory["sec_per_sample"])
    if budget < 0.4:
        k = min(k, K_MIN)
    return max(K_MIN, min(cfg["k_max"], k))


def decide(obs, memory) -> Action:
    t0 = time.perf_counter()
    model = memory["model"]

    # --- new hand / history bookkeeping
    if memory["cur_hand"] != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["n_seen"] = 0
        memory["my_last"] = None
    events = OM.parse_history(obs)
    for ev in events[memory["n_seen"]:]:
        if not ev[1]:
            OM.record(model, ev[2], ev[3])
    memory["n_seen"] = len(events)
    memory["cur_events"] = events

    legal = obs.legal_actions
    street_n = len(obs.board)

    bpm = int(memory["cfg"]["bp_mode"])
    if bpm and _BP is not None and _BP.AVAILABLE:
        use = ((bpm & 1) and obs.street == Street.PREFLOP) or \
              ((bpm & 2) and obs.hand_index < memory["cfg"]["bp_hands"])
        if use:
            best = _BP.act(obs, memory["rng"])
            if best in legal:
                return _finish(obs, memory, events, best, t0, 1)

    # --- simulate
    k = _choose_k(obs, memory)
    need = sorted({BOARD_N[Street(e[0])] for e in events if not e[1] and e[0] != Street.PREFLOP}
                  | ({street_n} if street_n else set()))
    sim = EQ.simulate(memory["rng"], obs.hole_cards, obs.board, k, need)
    TAU = cfg_tau = memory["cfg"]["tau"]
    w = _range_weights(events, sim, model, TAU)
    E = EQ.weighted_mean(sim.outcome, w)

    P, C = obs.pot, obs.to_call
    mb, ob, ms, os_ = obs.my_bet, obs.opp_bet, obs.my_stack, obs.opp_stack
    river = obs.street == Street.RIVER
    in_pos = obs.is_button if obs.street != Street.PREFLOP else not obs.is_button
    cfg = memory["cfg"]
    rf = 1.0 if river else (cfg["rf_ip"] if in_pos else cfg["rf_oop"])
    mode = int(cfg["rf_mode"])

    evs = {}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0
    c = min(C, ms)
    rf_call = 1.0 if (os_ == 0 or c >= ms) else rf
    evs[Action.CHECK_CALL] = _real(E, rf_call, mode) * (P + c) - c

    raises = [a for a in (Action.RAISE_HALF, Action.RAISE_POT) if a in legal]
    if raises:
        # How THIS opponent responds to a bet, conditioned on the range they
        # can still hold. Global frequencies give absolute strength
        # thresholds (fold below thr_f, re-raise above thr_r); the range
        # weights w then say how much of their current range is below/above.
        g = OM.group_of(obs.street)
        f_glob = OM.fold_freq(model, g)
        r_glob = OM.freq(model, (g, "facing"), "raise")
        zombie = OM.is_zombie(model)
        s_now = _strength(sim, street_n)
        lam = OM.reliability(model, g)
        p_cont = (1.0 - lam) * (1.0 - f_glob) + lam * _sig((s_now - f_glob) / TAU)
        p_rr = (1.0 - lam) * r_glob + lam * _sig((s_now - (1.0 - r_glob)) / TAU)
        p_rr = np.minimum(p_rr, p_cont)
        sw = max(w.sum(), 1e-12)
        f = 1.0 - float((w * p_cont).sum() / sw)
        r_all = float((w * p_rr).sum() / sw)
        if zombie:
            f, r_all = 0.97, 0.0
        w_call = w * (p_cont - p_rr)
        w_rr = w * p_rr
        E_c = EQ.weighted_mean(sim.outcome, w_call) if w_call.sum() > 1e-9 else E
        E_rr = EQ.weighted_mean(sim.outcome, w_rr) if w_rr.sum() > 1e-9 else E
        # if they can't re-raise, the would-be re-raisers just call
        w_c_only = w * p_cont
        E_c_norr = EQ.weighted_mean(sim.outcome, w_c_only) if w_c_only.sum() > 1e-9 else E
        for act in raises:
            r_to = obs.raise_to[act]
            a = r_to - mb
            b = r_to - ob
            P1 = P + a
            P2 = P1 + b
            all_in = a >= ms
            rf_c = 1.0 if all_in else rf
            can_rr = (not all_in) and obs.raises_this_street + 1 < 4
            rr = r_all if can_rr else 0.0
            called = _real(E_c if can_rr else E_c_norr, rf_c, mode) * P2 - a
            if rr > 0:
                r2 = min(r_to + P2, ob + os_)
                x = r2 - r_to
                P3 = P1 + (r2 - ob) + x
                rf_rr = 1.0 if (r2 >= ob + os_) else rf
                ev_rr = max(-a, _real(E_rr, rf_rr, mode) * P3 - a - x)
            else:
                ev_rr = 0.0
            ff = min(f, 1.0 - rr)
            evs[act] = ff * P + (1.0 - ff - rr) * called + rr * ev_rr

    best = max(evs, key=evs.get)
    if best not in legal:
        best = Action.CHECK_CALL

    use_hybrid = bool(bpm & 4)
    margin = memory["cfg"]["bp_margin"]
    if bpm & 8:
        reasons = OM.exploit_reasons(model)
        memory["exploiting"] = bool(reasons)
        use_hybrid = True
        if not reasons:
            margin = 1e9                      # balanced opponent: pure blueprint
    if use_hybrid and _BP is not None and _BP.AVAILABLE and not OM.is_zombie(model):
        ehs = float(sim.outcome.mean())
        made = sim.my_made if sim.my_made is not None else ehs
        if obs.street == Street.PREFLOP:
            ehs = made = EQ.preflop_pct(obs.hole_cards)
        node = _BP.node_of(obs)
        if node is not None:
            b = int(_BP.BK.bucket(int(obs.street), ehs, made, _BP.EDGES))
            if _BP.REACH[node, b] >= memory["cfg"]["bp_min_reach"]:
                p = _BP.probs(obs, memory["rng"], feats=(ehs, made))
                ev_bp = sum(p[a] * evs.get(a, 0.0) for a in p)
                memory["hyb"] = memory.get("hyb", 0) + 1
                if evs[best] - ev_bp <= margin * P:
                    acts = list(p)
                    best = acts[int(memory["rng"].choice(len(acts), p=np.array([p[a] for a in acts])))]
                else:
                    memory["hyb_exp"] = memory.get("hyb_exp", 0) + 1
    return _finish(obs, memory, events, best, t0, k)


def _finish(obs, memory, events, best, t0, k):
    memory["my_last"] = (int(obs.street), _kind(best, obs.to_call),
                         len([e for e in events if e[0] == obs.street]) == 0)
    memory["decisions"] += 1
    dt = time.perf_counter() - t0
    memory["spent"] += dt
    if k > 1:
        memory["sec_per_sample"] = 0.9 * memory["sec_per_sample"] + 0.1 * (dt / k)
    return best


def _kind(action: Action, to_call: int) -> str:
    if action == Action.FOLD:
        return "fold"
    if action == Action.CHECK_CALL:
        return "call" if to_call > 0 else "check"
    return "raise"


# ---------------------------------------------------------------------------

def hand_end(result, memory) -> None:
    t0 = time.perf_counter()
    model = memory["model"]
    if memory["cur_hand"] != result.hand_index:
        # we never acted: the small blind (opponent) folded straight away
        if result.winner == "me" and not result.went_to_showdown:
            OM.record(model, ("pre", "sb"), "fold")
        OM.decay(model)
        memory["spent"] += time.perf_counter() - t0
        return

    last = memory["my_last"]
    if last is not None:
        street, kind, first_on_street = last
        g = OM.group_of(street)
        if kind == "raise":
            if result.went_to_showdown:
                OM.record(model, (g, "facing"), "call")
            elif result.winner == "me":
                OM.record(model, (g, "facing"), "fold")
        elif kind == "check" and first_on_street and street == Street.RIVER and result.went_to_showdown:
            OM.record(model, ("post", "open"), "check")

    if result.went_to_showdown and result.opp_hole:
        _calibrate(result, memory)
    OM.decay(model)
    memory["spent"] += time.perf_counter() - t0


def _calibrate(result, memory) -> None:
    """At showdown: how strong were the hands the opponent raised with?"""
    model = memory["model"]
    opp = np.asarray(result.opp_hole, dtype=np.int64)
    board = list(result.board)
    for street, is_me, key, kind in memory["cur_events"]:
        if is_me or kind != "raise":
            continue
        if street == Street.PREFLOP:
            OM.calibrate(model, "pre", EQ.preflop_pct(opp))
        else:
            n = BOARD_N[Street(street)]
            vis = board[:n]
            dead = np.zeros(36, dtype=bool)
            dead[opp] = True
            dead[vis] = True
            rest = np.flatnonzero(~dead)
            rng = memory["rng"]
            others = rest[np.argsort(rng.random((200, len(rest))), axis=1)[:, :5]]
            from . import tables as T
            theirs = EQ._made_rank(T.hole_pair_idx(opp)[None, :], vis)[0]
            ranks = EQ._made_rank(T.batch_pair_idx(others), vis)
            pct = ((ranks < theirs).sum() + 0.5 * (ranks == theirs).sum()) / 200.0
            OM.calibrate(model, "post", float(pct))


def warm() -> None:
    """Touch the lookup tables and numpy code paths once at server start,
    so the first real decision isn't slow."""
    rng = np.random.default_rng(0)
    hole = [0, 5, 10, 15, 20]
    board = [25, 30, 35, 1, 6]
    for n in (0, 3, 4, 5):
        EQ.simulate(rng, hole, board[:n], 256, [x for x in (3, 4, 5) if x <= n])
    _ = float(EQ.T.PT.sum()) + float(EQ.PF_PCT.sum())
