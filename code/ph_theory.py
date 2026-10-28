"""First-order (single-tie) marginal value of conciliation, compared with simulation (run on h20).

For every round t = 1..T-1 of games played under 'conciliate throughout' (s = T-1), after the
round-t choices we compute, for every cooperator-defector pair (i, j), the exact one-step change in
the two endpoints' next-round cooperation probabilities caused by the tie's presence
    Delta_ij = [P_j(+ij) - P_j(-ij)] + [P_i(+ij) - P_i(-ij)],
holding all other ties at their current state, split into the fitted-rule part and the imitation
part.  Conciliating instead of excluding raises the tie's presence probability by
    q_ij = phi_add_bar (pair currently unlinked) or phi_del_bar (pair currently linked),
so the predicted myopic effect on next-round cooperation is M_t = (1/n) sum_ij q_ij Delta_ij.
M_{T-1} is compared with the simulated end-game difference coop(s=T-1) - coop(s=T-2).
Usage: python3 ph_theory.py <outdir> <nworkers>
"""
import sys, os, time, itertools, json
import numpy as np
import multiprocessing as mp
from numba import njit
import ph_sim as S


@njit(cache=True)
def _pfit(deg, xn, kk, th, betas):
    xr = xn / deg if deg > 0 else 0.0
    return 1.0 / (1.0 + np.exp(-(betas[2] + betas[3] * deg + kk * (betas[4] * xn + betas[5] * xr) + th)))


@njit(cache=True)
def _pimit(i, A, a, pay, tau, n, extra, include_extra, drop):
    """Imitation cooperation probability of i over neighbour set N_i (minus `drop`) plus optional `extra`."""
    tot = 0.0
    cnt = 0
    for l in range(n):
        if A[i, l] and l != drop:
            F = 1.0 / (1.0 + np.exp(-(pay[l] - pay[i]) / tau))
            tot += F * a[l] + (1 - F) * a[i]
            cnt += 1
    if include_extra:
        l = extra
        F = 1.0 / (1.0 + np.exp(-(pay[l] - pay[i]) / tau))
        tot += F * a[l] + (1 - F) * a[i]
        cnt += 1
    if cnt == 0:
        return -1.0
    return tot / cnt


@njit(cache=True)
def probe(seed, n, p, T, kappa, w, tau, mu_th, sd_th, phi, eps, D0, betas):
    """Play one donation game under conciliation throughout; return per-round terms (T-1, 8):
    [M_fit_gain(j), M_fit_loss(i), M_imit_gain(j), M_imit_loss(i), nCD, mean v_j, mean v_i, coop(t)]
    (sums over CD pairs weighted by q_ij, divided by n) plus pair-level means for the mean-field formula."""
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
    a = np.zeros(n, np.int64)
    a_prev = np.zeros(n, np.int64)
    pay_prev = np.zeros(n)
    pay = np.zeros(n)
    pc = np.zeros(n)
    out = np.zeros((T - 1, 12))
    qadd = 0.5 * (phi[2] + phi[3])
    qdel = 0.5 * (phi[0] + phi[1])
    for t in range(1, T + 1):
        deg = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if A[i, j]:
                    deg[i] += 1
        if t == 1:
            for i in range(n):
                pc[i] = 1.0 / (1.0 + np.exp(-(betas[0] + betas[1] * theta[i])))
        else:
            for i in range(n):
                xn = 0.0
                for j in range(n):
                    if A[i, j] and a_prev[j] == 1:
                        xn += 1
                kk = 1.0 if a_prev[i] == 1 else kappa
                pc[i] = _pfit(deg[i], xn, kk, theta[i], betas)
        for i in range(n):
            a[i] = 1 if np.random.random() < pc[i] else 0
        if t > 1 and w > 0:
            for i in range(n):
                if np.random.random() < w:
                    if deg[i] == 0:
                        continue
                    r = np.random.randint(int(deg[i]))
                    jj = -1
                    cnt = 0
                    for j in range(n):
                        if A[i, j]:
                            if cnt == r:
                                jj = j
                                break
                            cnt += 1
                    if np.random.random() < 1.0 / (1.0 + np.exp(-(pay_prev[jj] - pay_prev[i]) / tau)):
                        a[i] = a_prev[jj]
                    else:
                        a[i] = a_prev[i]
        for i in range(n):
            if d[i] - eps * deg[i] < 0:
                a[i] = 0
        nc = 0
        for i in range(n):
            nc += a[i]
            pi = 0.0
            for j in range(n):
                if A[i, j]:
                    pi += (1.0 if a[j] == 1 else 0.0) * 2.0 - (1.0 if a[i] == 1 else 0.0)
            pay[i] = eps * pi
            d[i] += pay[i]
        if t == T:
            break
        # ---- probe: marginal value of each C-D tie for next-round cooperation ----
        row = t - 1
        out[row, 7] = nc / n
        for i in range(n):
            for j in range(n):
                if i == j or a[i] != 1 or a[j] != 0:
                    continue
                # i cooperator, j defector
                linked = A[i, j]
                q = qdel if linked else qadd
                # degrees/xn without the tie
                di = deg[i] - (1 if linked else 0)
                dj = deg[j] - (1 if linked else 0)
                xi = 0.0
                xj = 0.0
                for l in range(n):
                    if l != j and A[i, l] and a[l] == 1:
                        xi += 1
                    if l != i and A[j, l] and a[l] == 1:
                        xj += 1
                pj0 = _pfit(dj, xj, kappa, theta[j], betas)
                pj1 = _pfit(dj + 1, xj + 1, kappa, theta[j], betas)
                pi0 = _pfit(di, xi, 1.0, theta[i], betas)
                pi1 = _pfit(di + 1, xi, 1.0, theta[i], betas)
                gj_fit = pj1 - pj0
                li_fit = pi1 - pi0
                # imitation parts (full P = (1-w) Pfit + w Pimit if deg > 0)
                mj0 = _pimit(j, A, a, pay, 0.1 if tau <= 0 else tau, n, i, False, i)
                mj1 = _pimit(j, A, a, pay, tau, n, i, True, i)
                mi0 = _pimit(i, A, a, pay, tau, n, j, False, j)
                mi1 = _pimit(i, A, a, pay, tau, n, j, True, j)
                Pj0 = pj0 if mj0 < 0 else (1 - w) * pj0 + w * mj0
                Pj1 = (1 - w) * pj1 + w * mj1
                Pi0 = pi0 if mi0 < 0 else (1 - w) * pi0 + w * mi0
                Pi1 = (1 - w) * pi1 + w * mi1
                gj_all = Pj1 - Pj0
                li_all = Pi1 - Pi0
                out[row, 0] += q * (1 - w) * gj_fit / n
                out[row, 1] += q * (1 - w) * li_fit / n
                out[row, 2] += q * (gj_all - (1 - w) * gj_fit) / n
                out[row, 3] += q * (li_all - (1 - w) * li_fit) / n
                out[row, 4] += 1.0
                out[row, 5] += pj0 * (1 - pj0)
                out[row, 6] += pi0 * (1 - pi0)
                out[row, 8] += dj
                out[row, 9] += di
                out[row, 10] += (xj / dj) if dj > 0 else 0.0
                out[row, 11] += (xi / di) if di > 0 else 0.0
        for i in range(n):
            a_prev[i] = a[i]
            pay_prev[i] = pay[i]
        # ---- planner: conciliate throughout (CC connect, DD cut, CD add/keep) ----
        for i in range(n):
            for j in range(i + 1, n):
                st = a[i] + a[j]
                v = 0
                if st >= 1 and not A[i, j]:
                    v = 1
                elif st == 0 and A[i, j]:
                    v = -1
                if v == 0:
                    continue
                ref = i if np.random.random() < 0.5 else j
                if v == -1:
                    ph = phi[0] if a[ref] == 0 else phi[1]
                else:
                    ph = phi[2] if a[ref] == 0 else phi[3]
                if np.random.random() < ph:
                    A[i, j] = v == 1
                    A[j, i] = v == 1
    return out


@njit(cache=True)
def probe_block(seeds, n, p, T, kappa, w, tau, mu_th, sd_th, phi, eps, D0, betas):
    acc = np.zeros((seeds.shape[0], T - 1, 12))
    for g in range(seeds.shape[0]):
        acc[g] = probe(seeds[g], n, p, T, kappa, w, tau, mu_th, sd_th, phi, eps, D0, betas)
    return acc


NG = 40000


def work(args):
    kap, w, pid = args
    P = S.DEFAULT
    T = 15
    seeds = np.arange(NG, dtype=np.int64) + 60000 * (3000 + pid)
    pr = probe_block(seeds, 16, 0.3, T, kap, w, 0.1, P['mu_th'], P['sd_th'], S.PHI0, 0.05, 1.0, P['betas'])
    M = pr[:, :, :4].mean(0)                          # (T-1, 4)
    Mtot = pr[:, :, :4].sum(2)
    Mse = Mtot.std(0, ddof=1) / np.sqrt(NG)
    npair = pr[:, :, 4].sum(0)
    means = pr[:, :, 5:12].sum(0) / np.maximum(npair[:, None], 1)
    means[:, 2] = pr[:, :, 7].mean(0)
    a = S.simulate(seeds, 0, T - 1, S.BIG, kappa=kap, w=w)
    b = S.simulate(seeds, 0, T - 2, S.BIG, kappa=kap, w=w)
    dd = a[:, 0] - b[:, 0]
    rows = []
    for t in range(T - 1):
        rows.append(dict(kappa=kap, w=w, t=t + 1, M_fit_gain=M[t, 0], M_fit_loss=M[t, 1], M_imit_gain=M[t, 2],
                         M_imit_loss=M[t, 3], M_total=M[t].sum(), M_se=Mse[t], pairs_per_game=npair[t] / NG,
                         v_j=means[t, 0], v_i=means[t, 1], coop_t=means[t, 2], d_j=means[t, 3], d_i=means[t, 4],
                         r_j=means[t, 5], r_i=means[t, 6],
                         sim_end_diff=float(dd.mean()) if t == T - 2 else np.nan,
                         sim_end_se=float(dd.std(ddof=1) / np.sqrt(NG)) if t == T - 2 else np.nan))
    return rows


if __name__ == '__main__':
    outdir, nw = sys.argv[1], int(sys.argv[2])
    if len(sys.argv) > 3:
        NG = 500
    os.makedirs(outdir, exist_ok=True)
    S.simulate(np.arange(4))
    probe_block(np.arange(2, dtype=np.int64), 16, 0.3, 15, 1.0, 0.0, 0.1, -0.304, 2.41, S.PHI0, 0.05, 1.0, S.DEFAULT['betas'])
    kaps = [round(x, 3) for x in np.linspace(0, 1.5, 16)]
    ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tasks = [(k, w, i) for i, (k, w) in enumerate(itertools.product(kaps, ws))]
    import pandas as pd
    allrows = []
    t0 = time.time()
    with mp.get_context('fork').Pool(nw) as pool:
        for rows in pool.imap_unordered(work, tasks):
            allrows += rows
    pd.DataFrame(allrows).sort_values(['kappa', 'w', 't']).to_csv(os.path.join(outdir, 'theory.csv'), index=False)
    print('theory done', time.time() - t0)
