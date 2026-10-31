"""Numerical checks of the revised Section 3, as reported in Appendix A of the thesis.

Model (Section 3): N workers, shard j owned by W_j; the owner's own gradient piece is always
counted (S >= 1, no fallback); a worker that misses a broadcast keeps its old copy.

  A1  exact enumeration of Theorem 3.1 (rational arithmetic)
  A2  Monte Carlo of Theorem 3.1 in the steady state: uncorrelated, constant, mixed and
      momentum updates
  A3  Corollary 3.2 (weights, unbiasedness, variance) and Corollary 3.6 (correlated senders)
  A4  full protocol, steps (1)-(4), with SGD on least squares: drift and Corollary 3.3
  A5  Section 3.3: Gilbert-Elliott across iterations, bursts within an iteration, interleaving
  A6  shard-level vs packet-level loss

(The thesis groups A1-A2 as Appendix A.1 and A3 as A.2; A4-A6 are A.3-A.5. Appendix A.6 uses
drop_log_mirror.py.)

Usage: python appendix_checks.py [A1 A2 ...]   (Python 3.8+, NumPy; fixed seeds; all checks
take a few minutes, A4 the longest)
"""
import itertools
import sys
from fractions import Fraction

import numpy as np


# ---------- closed forms of Theorem 3.1 (units of sigma_j^2) ----------
def pair_unc(p, t=None):            # (ii)
    if t is None:
        return 2 * p / (1 - p * p)
    return 2 * (p * (1 - p ** t) / (1 - p) - p * p * (1 - p ** (2 * t)) / (1 - p * p))


def pair_gen(p):                    # (i)
    return 2 * p / (1 - p) ** 2


def own_unc(p, t=None):             # (iii), uncorrelated
    return p / (1 - p) if t is None else p * (1 - p ** t) / (1 - p)


def own_gen(p):                     # (iii), general
    return p * (1 + p) / (1 - p) ** 2


def w_owner(N, p):                  # Corollary 3.2 (i)
    return (1 - p ** N) / (N * (1 - p))


# ---------- A1 ----------
def a1(p=Fraction(2, 5), T=8):
    """Enumerate the 2^(2t) delivery patterns of two non-owners W_i, W_k.

    Copies are tracked as 0/1 coefficient vectors over the updates (dtheta_0..dtheta_{t-1}), so
    E||D_t||^2 is exact for uncorrelated unit updates (sum of squared coefficients) and for
    constant unit updates (squared sum of coefficients).
    """
    print("A1  exact enumeration, p =", p)
    print(" t | pair unc. = (ii) | pair const <= (i) | owner unc. = (iii) | owner const <= (iii)")
    prev = None
    for t in range(1, T + 1):
        pu = pc = ou = oc = Fraction(0)
        for pat in itertools.product((0, 1), repeat=2 * t):
            ri, rk = pat[:t], pat[t:]
            prob = Fraction(1)
            for r in pat:
                prob *= (1 - p) if r else p
            owner, ci, ck = [0] * t, [0] * t, [0] * t
            for u in range(t):
                owner = owner.copy()
                owner[u] = 1                                  # theta_{u+1} = theta_u + dtheta_u
                ci = owner if ri[u] else ci                   # step (4)
                ck = owner if rk[u] else ck
            d = [a - b for a, b in zip(ci, ck)]
            e = [a - b for a, b in zip(owner, ci)]
            pu += prob * sum(x * x for x in d)
            pc += prob * sum(d) ** 2
            ou += prob * sum(x * x for x in e)
            oc += prob * sum(e) ** 2
        assert pu == pair_unc(p, t) and ou == own_unc(p, t), t      # exact rational equality
        assert pc <= pair_gen(p) and oc <= own_gen(p), t
        if prev:
            assert pc >= prev[0] and oc >= prev[1], t                # increasing in t
        prev = (pc, oc)
        print(f"{t:2d} | {float(pu):16.4f} | {float(pc):17.4f} | {float(ou):18.4f} | {float(oc):19.4f}")
    print(f" limits: (ii) {float(pair_unc(p)):.4f}  (i) {float(pair_gen(p)):.4f}  "
          f"(iii) {float(own_unc(p)):.4f} / {float(own_gen(p)):.4f}")
    print(" all equalities hold exactly as rational numbers; bounds hold and are approached\n")


# ---------- A2 ----------
def window_sum(m, rho):
    """sum_{u,v in a window of m consecutive updates} rho(|u - v|)."""
    return m + 2 * sum((m - k) * rho(k) for k in range(1, m))


def pred_stationary(p, rho, M=200):
    """Steady-state drift (units of sigma^2) for stationary updates whose correlation at lag k is
    rho(k): E[window_sum(L, rho)] with L = |G_i - G_k| (pair) or L = G (owner copy), G geometric,
    P(G = a) = (1 - p) p^a, so P(|G_i - G_k| = m) = 2 (1 - p) p^m / (1 + p) for m >= 1."""
    pair = sum(2 * (1 - p) / (1 + p) * p ** m * window_sum(m, rho) for m in range(1, M))
    own = sum((1 - p) * p ** a * window_sum(a, rho) for a in range(1, M))
    return pair, own


def a2(R=200_000, T=200, burn=100, beta=0.9):
    """Owner copy and two non-owner copies, steady state.

    Updates mu + s*xi with mu^2 + s^2 = 1; prediction (window argument of the proof):
    pair = mu^2 * (i) + s^2 * (ii), owner = mu^2 * (iii, general) + s^2 * (iii, uncorrelated).
    Momentum: updates m_t = beta m_{t-1} + (1 - beta) xi_t (how AdamW forms its update from
    uncorrelated gradients), correlation beta^k at lag k; prediction from pred_stationary.
    """
    print("A2  Monte Carlo, steady state (t in [100, 200), R = 200k)")
    models = [("uncorrelated", 0.0, 1.0), ("constant", 1.0, 0.0), ("mixed", 0.6, 0.8),
              (f"momentum, beta = {beta}", None, None)]
    for name, mu, s in models:
        print(f" updates {name}" + ("" if mu is None else f" (mu={mu}, s={s})"))
        for p in (0.1, 0.2, 0.3, 0.4):
            rng = np.random.default_rng(1)
            th = np.zeros(R)
            xi, xk = np.zeros(R), np.zeros(R)
            pair = own = sig = 0.0
            if mu is None:                                    # stationary start of the momentum
                mom = np.sqrt((1 - beta) / (1 + beta)) * rng.standard_normal(R)
            for t in range(T):
                if mu is None:
                    mom = beta * mom + (1 - beta) * rng.standard_normal(R)
                    d = mom
                else:
                    d = mu + s * rng.standard_normal(R)
                th = th + d
                xi = np.where(rng.random(R) >= p, th, xi)
                xk = np.where(rng.random(R) >= p, th, xk)
                if t >= burn:
                    pair += np.mean((xi - xk) ** 2)
                    own += np.mean((th - xi) ** 2)
                    sig += np.mean(d ** 2)
            pair, own = pair / sig, own / sig
            if mu is None:
                pp, po = pred_stationary(p, lambda k: beta ** k)
            else:
                pp = mu ** 2 * pair_gen(p) + s ** 2 * pair_unc(p)
                po = mu ** 2 * own_gen(p) + s ** 2 * own_unc(p)
            print(f"  p={p:.1f}  pair {pair:.4f} (pred {pp:.4f})   owner {own:.4f} (pred {po:.4f})")
    print()


# ---------- A3 ----------
def a3(N=8, R=2_000_000):
    print(f"A3  aggregation, N = {N}, R = {R:,}, owner = worker 0, sigma_g = 1")
    print("   p   | w_j formula  MC     | var. factor formula  MC   | |E g - mu| (mu_i = 1)")
    for p in (0.01, 0.1, 0.2, 0.3, 0.4):
        rng = np.random.default_rng(2)
        s = rng.random((R, N)) >= p
        s[:, 0] = True
        S = s.sum(1)
        g = 1.0 + rng.standard_normal((R, N))
        gh = (s * g).sum(1) / S
        wj = np.mean(1 / S)
        vf = N * gh.var()
        print(f"  {p:.2f} | {w_owner(N, p):.4f}  {wj:.4f}   | {N * w_owner(N, p):.4f}   {vf:.4f}   | "
              f"{abs(gh.mean() - 1):.1e}")
    # (i) with different expected pieces (stale replicas): mu_i = i
    p = 0.4
    rng = np.random.default_rng(3)
    s = rng.random((R, N)) >= p
    s[:, 0] = True
    mu = np.arange(N, dtype=float)
    gh = (s * (mu + rng.standard_normal((R, N)))).sum(1) / s.sum(1)
    wj = w_owner(N, p)
    pred = wj * mu[0] + (1 - wj) / (N - 1) * mu[1:].sum()
    print(f"  (i) mu_i = i, p = 0.4: E g = {gh.mean():.4f}, predicted {pred:.4f}, lossless mean {mu.mean():.4f}")
    # Corollary 3.6: losses correlated across senders (all non-owner pieces lost together w.p. p)
    lost = rng.random(R) < p
    s = np.broadcast_to(~lost[:, None], (R, N)).copy()
    s[:, 0] = True
    S = s.sum(1)
    gh = (s * (1.0 + rng.standard_normal((R, N)))).sum(1) / S
    print(f"  Cor 3.6, all-or-nothing loss p = 0.4: |E g - mu| = {abs(gh.mean() - 1):.1e}, "
          f"var. factor {N * gh.var():.4f}, predicted N E[1/S] = {N * (p + (1 - p) / N):.4f} "
          f"(independent: {N * w_owner(N, p):.4f})\n")


# ---------- A4 ----------
N4, D4 = 8, 64
SH4 = D4 // N4
OWNER = np.repeat(np.arange(N4), SH4)
COORD = np.arange(D4)
PAIRS = [[(i, k) for i in range(N4) for k in range(i + 1, N4) if j not in (i, k)] for j in range(N4)]


def problem(seed=1, n=4096, noise=0.5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, D4))
    b = A @ rng.standard_normal(D4) + noise * rng.standard_normal(n)
    H, c = A.T @ A / n, A.T @ b / n
    return A, b, H, c, np.linalg.solve(H, c)


def protocol(pg, pt, theta0, prob, *, eta, B, T, R, windows, seed=0):
    """Steps (1)-(4) of Section 3 with SGD; R independent runs. Returns per-window averages."""
    A, b, H, c, _ = prob
    ell = np.linalg.eigvalsh(H).max()                         # Lipschitz constant of grad L
    rng = np.random.default_rng(seed)
    rep = np.broadcast_to(theta0, (R, N4, D4)).copy()          # rep[r, i] = replica of W_i
    eye = np.eye(N4, dtype=bool)
    wj = w_owner(N4, pg)
    W = np.where(np.arange(N4)[:, None] == OWNER[None, :], wj, (1 - wj) / (N4 - 1))   # (N, D)
    out = [dict.fromkeys(("sig2", "mu2", "lag1", "pair", "own", "bias", "bound1", "viol"), 0.0) for _ in windows]
    dprev = np.zeros((R, N4, SH4))
    for t in range(T):
        theta = rep[:, OWNER, COORD]                          # reference model
        # Corollary 3.3 at the start of iteration t: E[g_hat | F_t] = sum_i w_i grad L(rep_i)
        grads = rep @ H - c                                   # (R, N, D)
        bias = ((W[None] * grads).sum(1) - (theta @ H - c)).reshape(R, N4, SH4)
        bias = np.linalg.norm(bias, axis=-1)                  # (R, shard)
        dist = np.linalg.norm(rep - theta[:, None, :], axis=-1)    # ||theta^[i] - theta||, (R, N)
        bound1 = ell * np.einsum("in,ri->rn", W[:, ::SH4], dist)    # first bound, per shard
        idx = rng.integers(0, len(b), size=(R, N4, B))
        Ab = A[idx]
        g = np.einsum("rnbd,rnb->rnd", Ab, np.einsum("rnbd,rnd->rnb", Ab, rep) - b[idx]) / B   # (1)
        s = rng.random((R, N4, N4)) >= pg                     # (2) s[r, i, j]: piece i -> j
        s |= eye
        sc = s[:, :, OWNER]
        ghat = (sc * g).sum(1) / sc.sum(1)
        theta_new = theta - eta * ghat                        # (3)
        rcv = rng.random((R, N4, N4)) >= pt                   # (4) rcv[r, j, i]: shard j -> W_i
        rcv |= eye
        rep = np.where(rcv[:, OWNER, :].transpose(0, 2, 1), theta_new[:, None, :], rep)
        dth = (theta_new - theta).reshape(R, N4, SH4)
        for (lo, hi), acc in zip(windows, out):
            if lo <= t < hi:
                acc["sig2"] += np.mean(np.sum(dth ** 2, -1))
                acc["mu2"] += np.mean(np.sum(dth.mean(0) ** 2, -1))
                acc["lag1"] += np.mean(np.sum(dth * dprev, -1))       # <dtheta_t, dtheta_{t-1}>, per shard
                x = rep.reshape(R, N4, N4, SH4)               # x[r, worker, shard]
                xo = theta_new.reshape(R, N4, SH4)
                acc["pair"] += np.mean([np.mean(np.sum((x[:, i, j] - x[:, k, j]) ** 2, -1))
                                        for j in range(N4) for i, k in PAIRS[j]])
                acc["own"] += np.mean([np.mean(np.sum((x[:, i, j] - xo[:, j]) ** 2, -1))
                                       for j in range(N4) for i in range(N4) if i != j])
                acc["bias"] += bias.mean()
                acc["bound1"] += bound1.mean()
                acc["viol"] = max(acc["viol"], float((bias - bound1 > 1e-9).mean()))
        dprev = dth
    res = []
    for (lo, hi), acc in zip(windows, out):
        m = hi - lo
        r = {k: (v / m if k != "viol" else v) for k, v in acc.items()}
        # second bound of Corollary 3.3 with sigma_m^2 = measured mean update size of shard m
        r["bound2"] = ell * np.sqrt(pt * (1 + pt)) / (1 - pt) * np.sqrt(N4 * r["sig2"])
        res.append(r)
    return res


def a4():
    prob = problem()
    th_ls = prob[4]
    print("A4  full protocol (N = 8, d = 64, least squares, SGD)")
    print(" stationary: start at the optimum, eta = 0.05, B = 16, R = 200; windows [500,1000), [2500,3000)")
    print("  p_g  p_th | pair/s2 early late  [(ii), (i)]     | owner/s2 early late [(iii)]      "
          "| E bias / bound1, bound2 | bound1 violated")
    for pg, pt in ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.4, 0.0), (0.0, 0.4)):
        e, l = protocol(pg, pt, th_ls, prob, eta=0.05, B=16, T=3000, R=200,
                        windows=[(500, 1000), (2500, 3000)])
        print(f"  {pg:.1f}  {pt:.1f} | {e['pair'] / e['sig2']:.4f} {l['pair'] / l['sig2']:.4f} "
              f"[{pair_unc(pt):.3f}, {pair_gen(pt):.3f}] | {e['own'] / e['sig2']:.4f} {l['own'] / l['sig2']:.4f} "
              f"[{own_unc(pt):.3f}, {own_gen(pt):.3f}] | "
              f"{l['bias'] / l['bound1'] if l['bound1'] else 0:.3f}, "
              f"{l['bias'] / l['bound2'] if l['bound2'] else 0:.3f} (E bias {l['bias']:.2e}) | {l['viol']:.3g}"
              f" | lag-1 corr {l['lag1'] / l['sig2'] if l['sig2'] else 0:+.4f}")
    z = np.random.default_rng(7).standard_normal(D4)
    print(" descent: start at optimum + 3z, eta = 0.002, B = 64, R = 400; window [45, 60)")
    print("  p | pair/s2  [(ii), (i)]   f = |E dth|^2/E|dth|^2  f(i)+(1-f)(ii) | owner/s2 [(iii)] "
          "| E bias / bound1, bound2")
    for p in (0.1, 0.2, 0.3, 0.4):
        (r,) = protocol(p, p, th_ls + 3 * z, prob, eta=0.002, B=64, T=60, R=400, windows=[(45, 60)])
        f = r["mu2"] / r["sig2"]
        print(f"  {p:.1f} | {r['pair'] / r['sig2']:.4f} [{pair_unc(p):.3f}, {pair_gen(p):.3f}]  {f:.3f}  "
              f"{f * pair_gen(p) + (1 - f) * pair_unc(p):.4f} | {r['own'] / r['sig2']:.4f} "
              f"[{own_unc(p):.3f}, {own_gen(p):.3f}] | {r['bias'] / r['bound1']:.3f}, {r['bias'] / r['bound2']:.3f}"
              f" | viol {r['viol']:.3g} | lag-1 corr {r['lag1'] / r['sig2']:+.4f}")
    print()


# ---------- A5 ----------
P_GB, P_BG, D_G, D_B = 0.01, 0.10, 0.01, 0.70                # example of Section 3.3
PI_B = P_GB / (P_GB + P_BG)
P_BAR = (1 - PI_B) * D_G + PI_B * D_B


def ge_moments():
    """Corollary 3.5: m1 = a L (I - P L)^-1 1, m2 = 2 a L (I - P L)^-2 1 - m1."""
    P = np.array([[1 - P_GB, P_GB], [P_BG, 1 - P_BG]])
    L = np.diag([D_G, D_B])
    a = np.array([1 - PI_B, PI_B])
    M = np.linalg.inv(np.eye(2) - P @ L)
    m1 = a @ L @ M @ np.ones(2)
    return m1, 2 * a @ L @ M @ M @ np.ones(2) - m1


def ge_states(rng, steps):
    """Stationary Gilbert-Elliott state path (True = Bad) of length `steps`, from sojourn times."""
    n = int(steps * (P_GB * P_BG / (P_GB + P_BG)) * 2.5) + 10
    good = rng.geometric(P_GB, n)
    bad = rng.geometric(P_BG, n)
    lens = np.empty(2 * n, np.int64)
    lens[0::2], lens[1::2] = good, bad
    st = np.repeat(np.tile([False, True], n), lens)
    off = rng.integers(0, 1000)                               # random phase
    assert len(st) >= steps + off
    return st[off:off + steps]


def ages(lost):
    """A_t = number of consecutive losses immediately before t, along axis 0 (A_0 = 0)."""
    T = lost.shape[0]
    t = np.arange(T, dtype=np.int32).reshape((T,) + (1,) * (lost.ndim - 1))
    last = np.maximum.accumulate(np.where(~lost, t, -1), axis=0)     # last delivery at or before t
    a = np.zeros(lost.shape, np.int32)
    a[1:] = t[1:] - 1 - last[:-1]
    return a


def a5():
    rng = np.random.default_rng(5)
    m1, m2 = ge_moments()
    print(f"A5  beyond i.i.d.; Gilbert-Elliott example: mean loss {P_BAR:.4f}, m1 = {m1:.4f}, m2 = {m2:.4f};"
          f" i.i.d. at the same rate: m1 = {P_BAR / (1 - P_BAR):.4f}, m2 = {own_gen(P_BAR):.4f}")
    # (a) one chain per link, one step per iteration; 2 x 64 links, uncorrelated and constant updates
    T, Wd, BURN = 200_000, 64, 1000
    lost = np.empty((T, 2, Wd), bool)
    for i in range(2):
        for w in range(Wd):
            st = ge_states(rng, T)
            lost[:, i, w] = rng.random(T) < np.where(st, D_B, D_G)
    A = ages(lost)[BURN:]
    th = np.zeros(Wd)
    e = np.zeros((2, Wd))                                     # random-walk staleness errors
    pu = ou = 0.0
    for t in range(T):
        d = rng.standard_normal(Wd)
        e = np.where(lost[t], e + d, 0.0)                     # e_{t+1} = (1 - r_t)(e_t + dtheta_t)
        if t >= BURN - 1 and t < T - 1:
            ou += np.mean(e[0] ** 2)
            pu += np.mean((e[0] - e[1]) ** 2)
    n = T - BURN
    print(" (a) GE across iterations, uncorrelated updates: owner {:.4f} (Thm 3.4(iii): <= m1 = {:.4f}), "
          "pair {:.4f} (<= 2 m1 = {:.4f})".format(ou / n, m1, pu / n, 2 * m1))
    print("     constant updates: owner E[A^2] {:.4f} (Thm 3.4(i): <= m2 = {:.4f}), pair E[(A_i-A_k)^2] {:.4f} "
          "(Thm 3.4(ii): <= 2 m2 = {:.4f}); E[A] {:.4f}; realized loss {:.4f}".format(
              np.mean(A[:, 0] ** 2), m2, np.mean((A[:, 0] - A[:, 1]) ** 2), 2 * m2, np.mean(A[:, 0]),
              lost.mean()))
    # (b) bursts within an iteration: a shard of n blocks (one per packet) per transfer
    n_blk, T = 1000, 20_000
    print(f" (b) {n_blk} blocks per transfer, {T} iterations, two receivers; per-block uncorrelated drift")
    print("     process          | loss  | owner E[A] | pair E|Ai-Ak| | pair, worst 10% of blocks")

    def process(name):
        if name == "iid":
            return rng.random((T, n_blk)) < P_BAR
        if name == "GE in transfer":                       # one chain through all packets; bursts ~10 packets
            st = ge_states(rng, T * n_blk).reshape(T, n_blk)
            return rng.random((T, n_blk)) < np.where(st, D_B, D_G)
        prob = np.where(np.arange(n_blk) < n_blk // 10, 0.45, 0.031)   # position-dependent (incast)
        drop = rng.random((T, n_blk)) < prob
        if name == "incast, interleaved":                  # random block -> packet map per iteration
            perm = np.argsort(rng.random((T, n_blk)), axis=1)
            drop = np.take_along_axis(drop, perm, axis=1)
        return drop

    for name in ("iid", "GE in transfer", "incast", "incast, interleaved"):
        l1, l2 = process(name), process(name)
        a1_, a2_ = ages(l1)[200:], ages(l2)[200:]
        pair_b = np.abs(a1_ - a2_).mean(0)
        print(f"     {name:16} | {l1.mean():.4f} | {a1_.mean():.4f}     | {pair_b.mean():.4f}        | "
              f"{np.sort(pair_b)[-n_blk // 10:].mean():.4f}")
    pbar = 0.1 * 0.45 + 0.9 * 0.031
    print(f"     i.i.d. prediction at p = {P_BAR:.4f}: owner {own_unc(P_BAR):.4f}, pair {pair_unc(P_BAR):.4f};"
          f" incast without interleaving (Jensen): owner {0.1 * own_unc(0.45) + 0.9 * own_unc(0.031):.4f},"
          f" pair {0.1 * pair_unc(0.45) + 0.9 * pair_unc(0.031):.4f}, worst 10% {pair_unc(0.45):.4f}"
          f" (incast mean {pbar:.4f})\n")


# ---------- A6 ----------
PAIRS6 = 50


def a6(N=8, C=200, T=4000, P=0.4, BURN=200):
    rng = np.random.default_rng(6)
    print(f"A6  shard- vs packet-level loss, N = {N}, {C} blocks per shard, p = {P}, {T} iterations, "
          f"{PAIRS6} receiver pairs")
    print("     unit   | pair drift/s2 | E(g_hat-mu)^2 | std stale fraction | std lost fraction (gradients)")
    for level in ("shard", "packet"):
        w = 1 if level == "shard" else C
        th, rep = np.zeros((PAIRS6, C)), np.zeros((2, PAIRS6, C))   # independent receiver pairs
        acc, stale = 0.0, []
        for t in range(T):
            th = th + rng.standard_normal((PAIRS6, C))
            rep = np.where(rng.random((2, PAIRS6, w)) >= P, th, rep)
            if t >= BURN:
                acc += np.mean((rep[0] - rep[1]) ** 2)
                stale.append(np.mean(rep[0] != th, axis=-1))
        stale = np.array(stale)
        err, frac = [], []
        for _ in range(T):
            s = np.broadcast_to(rng.random((N, w)) >= P, (N, C)).copy()
            s[0] = True
            g = 1.0 + rng.standard_normal((N, C))
            err.append(np.mean(((s * g).sum(0) / s.sum(0) - 1.0) ** 2))
            frac.append(1 - s[1:].mean())
        print(f"     {level:6} | {acc / (T - BURN):13.4f} | {np.mean(err):13.4f} | {stale.std(0).mean():18.4f} | "
              f"{np.std(frac):.4f}")
    q = P * (1 - P)
    print(f"     theory | {pair_unc(P):13.4f} | {w_owner(N, P):13.4f} | shard {np.sqrt(q):.4f}, packet "
          f"{np.sqrt(q / C):.4f} | shard {np.sqrt(q / (N - 1)):.4f}, packet {np.sqrt(q / ((N - 1) * C)):.4f}\n")


if __name__ == "__main__":
    checks = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6}
    for k in sys.argv[1:] or checks:
        checks[k]()
