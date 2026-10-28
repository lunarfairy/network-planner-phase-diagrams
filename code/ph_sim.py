"""Fast (numba) simulator of planner-steered adaptive networks for the phase-diagram study.

Faithful port of sim_pilot.py (McKee et al. 2023 game + fitted bot model, SI Tables 1, 3, 7-10)
with three generalisations used as phase-diagram axes:
  * behavioural: defector responsiveness kappa, payoff-imitation share w (Fermi, temperature tau),
    disposition mean/spread (mu_th, sd_th), the four acceptance probabilities phi, and an
    alternative 'unanimity' response rule;
  * game: a general symmetric 2x2 game per tie, payoff eps*[R, S, T, P] (donation b/c=2 is
    R=1, S=-1, T=2, P=0 with eps=0.05, i.e. b=0.10, c=0.05);
  * structural: group size n, initial tie density p, horizon T.

Planner codes
  0  family (s, k): C-C connect, D-D cut; a C-D pair is conciliated (add / keep) in round t
     iff t <= s and the defector's cumulative number of defections m <= k; otherwise excluded
     (delete if present).  s = 0 is pure exclusion; s = T-1, k = BIG is conciliation throughout;
     0 < s < T-1, k = BIG is the clock-switch ('conciliate-then-exclude') policy; k < BIG is
     history-conditioned probation.
  1 static   2 random (p=0.3 toggles)   3 max-connectivity   4 neutral table
  5 encouragement table (published distillation of the GraphNet planner)   6 cooperative clustering
"""
import numpy as np
from numba import njit

BIG = 1000
B0 = (1.807, 0.818)
BL = (-0.010, -0.193, 0.370, 1.521)

ENC_CC = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, .010], [1, .010],
                   [1, .011], [1, .028], [.991, .035], [.954, .073], [1, .108]])
ENC_CD = np.array([[.993, .048], [.973, .029], [.914, .145], [.791, .213], [.644, .318], [.594, .508],
                   [.463, .608], [.429, .745], [.366, .802], [.372, .753], [.361, .741], [.371, .774],
                   [.328, .706], [.408, .722]])
NEUTRAL = np.array([[.891, .119], [.841, .054], [.656, .084], [.642, .102], [.608, .117], [.549, .204],
                    [.545, .215], [.538, .224], [.520, .239], [.504, .213], [.532, .215], [.518, .237],
                    [.529, .232], [.522, .317]])
# phi order: [del|ref D, del|ref C, add|ref D, add|ref C]
PHI0 = np.array([0.774, 0.085, 0.287, 0.909])
DONATION = np.array([1.0, -1.0, 2.0, 0.0])   # R, S, T, P in units of eps


@njit(cache=True)
def _lg(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def _table_row(tab, t):
    # tables have 14 rows for t = 1..14; longer horizons reuse the last row
    r = t - 1
    if r > tab.shape[0] - 1:
        r = tab.shape[0] - 1
    return r


@njit(cache=True)
def play(seed, code, s, k, n, p, T, kappa, w, tau, mu_th, sd_th, phi, game, eps, D0, behaviour,
         enc_cc, enc_cd, neutral, betas, traj, branch):
    """One game. Returns (final coop, mean coop, mean capital, capital gini, final density, recs/round).
    betas = [b0_int, b0_theta, bl_int, bl_s, bl_n, bl_r(, round trend)].
    If traj has T rows, per-round diagnostics are ADDED to it (no random numbers are drawn, so the
    random stream and all outcomes are identical to a run without recording):
      0 cooperation, 1 cooperation of players who defected in round `branch`, 2 cooperation of
      players who cooperated in round `branch` (both 0 for rounds <= branch), 3 mean degree,
      4 number of C-D ties, 5 number of capital-forced defections, 6 n defectors in round `branch`,
      7 n cooperators in round `branch`."""
    np.random.seed(seed)
    A = np.zeros((n, n), np.bool_)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                A[i, j] = True
                A[j, i] = True
    theta = np.empty(n)
    for i in range(n):
        theta[i] = mu_th + sd_th * np.random.normal()
    d = np.full(n, D0)
    ndef = np.zeros(n)
    a = np.zeros(n, np.int64)
    a_prev = np.zeros(n, np.int64)
    pay_prev = np.zeros(n)
    pay = np.zeros(n)
    pc = np.zeros(n)
    coop_sum = 0.0
    final = 0.0
    nrec_tot = 0
    disc_cap = 0.0       # sum_t 0.99^(t-1) * mean capital after round t (original planner reward, no penalty)
    disc_pen = 0.0       # sum over planner steps of 0.99^(t-1) * rejected recommendations / n_pairs
    npairs = n * (n - 1) / 2.0
    Rg, Sg, Tg, Pg = game[0], game[1], game[2], game[3]
    worst_c = min(Rg, Sg) * eps       # worst-case payoff per tie of cooperating
    focal = np.empty(5, np.int64)
    if code == 6:
        perm = np.random.permutation(n)
        for q in range(5):
            focal[q] = perm[q]
    Rm = np.zeros((n, n), np.int64)
    rec = traj.shape[0] >= T
    grpD = np.zeros(n, np.bool_)
    for t in range(1, T + 1):
        deg = np.zeros(n)
        for i in range(n):
            c_ = 0
            for j in range(n):
                if A[i, j]:
                    c_ += 1
            deg[i] = c_
        if t == 1:
            for i in range(n):
                pc[i] = _lg(betas[0] + betas[1] * theta[i])
        else:
            for i in range(n):
                xn = 0.0
                for j in range(n):
                    if A[i, j] and a_prev[j] == 1:
                        xn += 1.0
                xr = xn / deg[i] if deg[i] > 0 else 0.0
                kk = 1.0 if a_prev[i] == 1 else kappa
                if behaviour == 0:
                    trend = betas[6] * (t - 1) if betas.shape[0] > 6 else 0.0   # optional round trend
                    pc[i] = _lg(betas[2] + betas[3] * deg[i] + kk * (betas[4] * xn + betas[5] * xr) + theta[i] + trend)
                else:  # unanimity rule
                    if deg[i] == 0:
                        pc[i] = 0.5
                    elif xn == deg[i]:
                        pc[i] = 0.9 if a_prev[i] == 1 else 0.7
                    else:
                        pc[i] = 0.02
        for i in range(n):
            a[i] = 1 if np.random.random() < pc[i] else 0
        if t > 1 and w > 0:
            for i in range(n):
                if np.random.random() < w:
                    if deg[i] == 0:
                        continue
                    # random neighbour
                    r = np.random.randint(int(deg[i]))
                    jj = -1
                    cnt = 0
                    for j in range(n):
                        if A[i, j]:
                            if cnt == r:
                                jj = j
                                break
                            cnt += 1
                    if np.random.random() < _lg((pay_prev[jj] - pay_prev[i]) / tau):
                        a[i] = a_prev[jj]
                    else:
                        a[i] = a_prev[i]
        nbind = 0
        for i in range(n):
            if d[i] + worst_c * deg[i] < 0:
                if a[i] == 1:
                    nbind += 1
                a[i] = 0
        nc = 0
        for i in range(n):
            nc += a[i]
        for i in range(n):
            pi = 0.0
            for j in range(n):
                if A[i, j]:
                    if a[i] == 1:
                        pi += Rg if a[j] == 1 else Sg
                    else:
                        pi += Tg if a[j] == 1 else Pg
            pay[i] = eps * pi
            d[i] += pay[i]
            if a[i] == 0:
                ndef[i] += 1.0
        frac = nc / n
        mc = 0.0
        for i in range(n):
            mc += d[i]
        disc_cap += 0.99 ** (t - 1) * mc / n
        if rec:
            if t == branch:
                for i in range(n):
                    grpD[i] = a[i] == 0
            nD = 0
            cD = 0
            cC = 0
            for i in range(n):
                if grpD[i]:
                    nD += 1
                    cD += a[i]
                else:
                    cC += a[i]
            ncd = 0
            sdeg = 0.0
            for i in range(n):
                sdeg += deg[i]
                for j in range(i + 1, n):
                    if A[i, j] and a[i] != a[j]:
                        ncd += 1
            traj[t - 1, 0] += frac
            if t > branch and branch > 0:
                traj[t - 1, 1] += cD / nD if nD > 0 else 0.0
                traj[t - 1, 2] += cC / (n - nD) if nD < n else 0.0
                traj[t - 1, 6] += nD
                traj[t - 1, 7] += n - nD
                traj[t - 1, 8] += cD / n
                traj[t - 1, 9] += cC / n
            traj[t - 1, 3] += sdeg / n
            traj[t - 1, 4] += ncd
            traj[t - 1, 5] += nbind
        coop_sum += frac
        final = frac
        for i in range(n):
            a_prev[i] = a[i]
            pay_prev[i] = pay[i]
        if t == T:
            break
        # ---------------- planner ----------------
        for i in range(n):
            for j in range(n):
                Rm[i, j] = 0
        if code == 0 or code == 4 or code == 5:
            for i in range(n):
                for j in range(i + 1, n):
                    st = a[i] + a[j]
                    if code == 0:
                        if st == 2:
                            padd, pdel = 1.0, 0.0
                        elif st == 0:
                            padd, pdel = 0.0, 1.0
                        else:
                            m = ndef[i] if a[i] == 0 else ndef[j]
                            if t <= s and m <= k:
                                padd, pdel = 1.0, 0.0
                            else:
                                padd, pdel = 0.0, 1.0
                    elif code == 4:
                        rr = _table_row(neutral, t)
                        padd, pdel = neutral[rr, 0], neutral[rr, 1]
                    else:
                        rr = _table_row(enc_cc, t)
                        if st == 2:
                            padd, pdel = enc_cc[rr, 0], enc_cc[rr, 1]
                        elif st == 1:
                            padd, pdel = enc_cd[rr, 0], enc_cd[rr, 1]
                        else:
                            padd, pdel = 0.0, 1.0
                    u = np.random.random()
                    if (not A[i, j]) and u < padd:
                        Rm[i, j] = 1
                    elif A[i, j] and u < pdel:
                        Rm[i, j] = -1
        elif code == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    if np.random.random() < 0.30:
                        Rm[i, j] = -1 if A[i, j] else 1
        elif code == 3:
            for i in range(n):
                for j in range(i + 1, n):
                    if not A[i, j]:
                        Rm[i, j] = 1
        elif code == 6:
            newf = np.empty(5, np.int64)
            for q in range(5):
                f = focal[q]
                if a[f] == 1:
                    nd = 0
                    for j in range(n):
                        if A[f, j] and a[j] == 0:
                            nd += 1
                    if nd > 0:
                        r = np.random.randint(nd)
                        cnt = 0
                        for j in range(n):
                            if A[f, j] and a[j] == 0:
                                if cnt == r:
                                    Rm[min(f, j), max(f, j)] = -1
                                    break
                                cnt += 1
                    else:
                        ncand = 0
                        for j in range(n):
                            if a[j] == 1 and (not A[f, j]) and j != f:
                                ncand += 1
                        if ncand > 0:
                            r = np.random.randint(ncand)
                            cnt = 0
                            for j in range(n):
                                if a[j] == 1 and (not A[f, j]) and j != f:
                                    if cnt == r:
                                        Rm[min(f, j), max(f, j)] = 1
                                        break
                                    cnt += 1
                    newf[q] = f
                else:
                    ncand = 0
                    for j in range(n):
                        if a[j] == 1:
                            ok = True
                            for q2 in range(5):
                                if focal[q2] == j:
                                    ok = False
                            for q2 in range(q):
                                if newf[q2] == j:
                                    ok = False
                            if ok:
                                ncand += 1
                    if ncand > 0:
                        r = np.random.randint(ncand)
                        cnt = 0
                        for j in range(n):
                            if a[j] == 1:
                                ok = True
                                for q2 in range(5):
                                    if focal[q2] == j:
                                        ok = False
                                for q2 in range(q):
                                    if newf[q2] == j:
                                        ok = False
                                if ok:
                                    if cnt == r:
                                        newf[q] = j
                                        break
                                    cnt += 1
                    else:
                        newf[q] = f
            for q in range(5):
                focal[q] = newf[q]
            for i in range(n):
                for j in range(i + 1, n):
                    if Rm[i, j] == 0 and np.random.random() < 0.05:
                        Rm[i, j] = -1 if A[i, j] else 1
        # ---------------- acceptance ----------------
        for i in range(n):
            for j in range(i + 1, n):
                v = Rm[i, j]
                if v == 0:
                    continue
                nrec_tot += 1
                ref = i if np.random.random() < 0.5 else j
                if v == -1:
                    ph = phi[0] if a[ref] == 0 else phi[1]
                else:
                    ph = phi[2] if a[ref] == 0 else phi[3]
                if np.random.random() < ph:
                    A[i, j] = v == 1
                    A[j, i] = v == 1
                else:
                    disc_pen += 0.99 ** (t - 1) / npairs
    # outcomes
    tot = 0.0
    for i in range(n):
        tot += d[i]
    xs = np.sort(d)
    gsum = 0.0
    for q in range(n):
        gsum += (2 * (q + 1) - n - 1) * xs[q]
    gini = gsum / (n * tot) if tot > 0 else 0.0
    ne = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ne += 1
    return final, coop_sum / T, tot / n, gini, ne / (n * (n - 1) / 2), nrec_tot / (T - 1), disc_cap, disc_cap - disc_pen


@njit(cache=True)
def run_block(seeds, code, s, k, n, p, T, kappa, w, tau, mu_th, sd_th, phi, game, eps, D0, behaviour,
              enc_cc, enc_cd, neutral, betas):
    out = np.empty((seeds.shape[0], 8))
    tr = np.zeros((0, 8))
    for g in range(seeds.shape[0]):
        r = play(seeds[g], code, s, k, n, p, T, kappa, w, tau, mu_th, sd_th, phi, game, eps, D0,
                 behaviour, enc_cc, enc_cd, neutral, betas, tr, 0)
        for q in range(8):
            out[g, q] = r[q]
    return out


@njit(cache=True)
def run_traj(seeds, code, s, k, n, p, T, kappa, w, tau, mu_th, sd_th, phi, game, eps, D0, behaviour,
             enc_cc, enc_cd, neutral, betas, branch):
    """Per-game outcomes plus per-round diagnostics summed over games (divide by len(seeds))."""
    out = np.empty((seeds.shape[0], 8))
    tr = np.zeros((T, 10))
    for g in range(seeds.shape[0]):
        r = play(seeds[g], code, s, k, n, p, T, kappa, w, tau, mu_th, sd_th, phi, game, eps, D0,
                 behaviour, enc_cc, enc_cd, neutral, betas, tr, branch)
        for q in range(8):
            out[g, q] = r[q]
    return out, tr


DEFAULT = dict(n=16, p=0.3, T=15, kappa=1.0, w=0.0, tau=0.1, mu_th=-0.304, sd_th=2.410,
               phi=PHI0, game=DONATION, eps=0.05, D0=1.0, behaviour=0,
               betas=np.array([B0[0], B0[1], BL[0], BL[1], BL[2], BL[3]]))

CODES = dict(family=0, static=1, random=2, maxconn=3, neutral=4, encouragement=5, clustering=6)


def simulate(seeds, code=0, s=0, k=BIG, **kw):
    P = dict(DEFAULT)
    P.update(kw)
    return run_block(np.asarray(seeds, np.int64), int(code), int(s), float(k), int(P['n']), float(P['p']),
                     int(P['T']), float(P['kappa']), float(P['w']), float(P['tau']), float(P['mu_th']),
                     float(P['sd_th']), np.asarray(P['phi'], float), np.asarray(P['game'], float),
                     float(P['eps']), float(P['D0']), int(P['behaviour']), ENC_CC, ENC_CD, NEUTRAL,
                     np.asarray(P['betas'], float))


def strategy_set(T, ks=(0, 1, 2, 4, BIG)):
    """(s, k) grid: exclusion (s=0) plus every switch round s = 1..T-1 crossed with the strike limits."""
    out = [(0, BIG)]
    for s in range(1, T):
        for k in ks:
            out.append((s, k))
    return out


def simulate_traj(seeds, code=0, s=0, k=BIG, branch=0, **kw):
    """As simulate(), also returning mean per-round diagnostics (see play)."""
    P = dict(DEFAULT)
    P.update(kw)
    out, tr = run_traj(np.asarray(seeds, np.int64), int(code), int(s), float(k), int(P['n']), float(P['p']),
                       int(P['T']), float(P['kappa']), float(P['w']), float(P['tau']), float(P['mu_th']),
                       float(P['sd_th']), np.asarray(P['phi'], float), np.asarray(P['game'], float),
                       float(P['eps']), float(P['D0']), int(P['behaviour']), ENC_CC, ENC_CD, NEUTRAL,
                       np.asarray(P['betas'], float), int(branch))
    return out, tr / len(seeds)


def distinct_rules(T):
    """Behaviourally distinct members of Pi(s, k).  The planner acts after round t on a C-D pair whose
    defector has m >= 1 cumulative defections (it defected in round t) and m <= t.  Hence k = 0 never
    conciliates (identical to exclusion) and k >= s is identical to the clock rule (s, BIG).  At T = 15
    this leaves 50 rules: exclusion, 14 clock rules and 35 probation rules with 1 <= k < s."""
    out = [(0, BIG)]
    for s_ in range(1, T):
        out.append((s_, BIG))
    for kk in (1, 2, 4):
        for s_ in range(kk + 1, T):
            out.append((s_, kk))
    return out
