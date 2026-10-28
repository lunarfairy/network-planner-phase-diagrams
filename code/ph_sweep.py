"""Two-stage strategy optimisation over phase-diagram grids (run on h20).

Stage 1: every (s, k) strategy, N1 games, common random numbers (CRN) across strategies.
Stage 2: candidates within 3 paired SE of the stage-1 best (<= 12) plus exclusion, conciliation
         throughout, the capital-best strategy and two reference planners, re-evaluated on N2 fresh
         CRN games.  Paired differences vs the stage-2 best and vs exclusion are stored.
Usage: python3 ph_sweep.py <outdir> <nworkers> [quick]
"""
import sys, os, time, json, itertools
import numpy as np
import multiprocessing as mp
import ph_sim as S

N1, N2 = 2000, 10000
KAPPAS = [round(x, 3) for x in np.linspace(0, 1.5, 16)]


def grids():
    pts = []
    for kap, w in itertools.product(KAPPAS, [round(x, 2) for x in np.arange(0, 0.61, 0.05)]):
        pts.append(dict(grid='kw', kappa=kap, w=w))
    for w in (0.0, 0.3, 0.6):
        for Tg, Sg in itertools.product(np.arange(0, 3.01, 0.25), np.arange(-2, 1.01, 0.25)):
            pts.append(dict(grid='game', kappa=1.0, w=w, game=[1.0, float(Sg), float(Tg), 0.0]))
    for kap, pad in itertools.product(KAPPAS, [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]):
        phi = S.PHI0.copy(); phi[2] = pad
        pts.append(dict(grid='phi', kappa=kap, phi=phi.tolist(), phi_AD=pad))
    for kap, T in itertools.product(KAPPAS, [5, 8, 10, 15, 20, 25, 30]):
        pts.append(dict(grid='horizon', kappa=kap, T=T))
    for kap, sc in itertools.product(KAPPAS, [0.25, 0.5, 1.0, 1.5, 2.0]):
        pts.append(dict(grid='sigma', kappa=kap, sd_th=2.410 * sc, sd_scale=sc))
    for kap, dm in itertools.product(KAPPAS, [-1.0, -0.5, 0.0, 0.5, 1.0]):
        pts.append(dict(grid='mu', kappa=kap, mu_th=-0.304 + dm, mu_shift=dm))
    for kap, n in itertools.product(KAPPAS, [8, 16, 32, 64]):
        pts.append(dict(grid='size', kappa=kap, n=n, p=4.5 / (n - 1)))
    for w in [round(x, 2) for x in np.arange(0, 0.61, 0.05)]:
        pts.append(dict(grid='unanimity', kappa=1.0, w=w, behaviour=1))
    for i, p in enumerate(pts):
        p['pid'] = i
    return pts


SIMKEYS = ('n', 'p', 'T', 'kappa', 'w', 'tau', 'mu_th', 'sd_th', 'phi', 'game', 'eps', 'D0', 'behaviour', 'betas')


def simkw(pt):
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in pt.items() if k in SIMKEYS}


def evaluate(pt, n1=None, n2=None):
    n1 = n1 or N1
    n2 = n2 or N2
    kw = simkw(pt)
    T = int(pt.get('T', 15))
    strat = S.strategy_set(T)
    seeds1 = np.arange(n1, dtype=np.int64) + 60000 * (2 * pt['pid'])      # uint32-safe seed blocks
    res1 = np.stack([S.simulate(seeds1, 0, s, k, **kw) for s, k in strat])      # (nstrat, n1, 6)
    fin, cap = res1[:, :, 0], res1[:, :, 2]
    ib = int(np.argmax(fin.mean(1)))
    dse = (fin - fin[ib]).std(1, ddof=1) / np.sqrt(n1)
    gap = fin[ib].mean() - fin.mean(1)
    order = np.argsort(-fin.mean(1))
    cand = [int(i) for i in order if gap[i] <= 3 * max(dse[i], 1e-9)][:12]
    icap = int(np.argmax(cap.mean(1)))
    iconc = strat.index((T - 1, S.BIG))
    for extra in (0, iconc, icap):
        if extra not in cand:
            cand.append(extra)
    stage1 = []
    for q, (s, k) in enumerate(strat):
        stage1.append(dict(pid=pt['pid'], s=s, k=k, fin=float(fin[q].mean()), fin_se=float(fin[q].std(ddof=1) / np.sqrt(n1)),
                           cap=float(cap[q].mean()), meanc=float(res1[q, :, 1].mean())))
    seeds2 = np.arange(n2, dtype=np.int64) + 60000 * (2 * pt['pid'] + 1)
    rows, outs = [], []
    labels = [('family', strat[c][0], strat[c][1]) for c in cand] + [('encouragement', -1, -1), ('static', -1, -1)]
    for lab, s, k in labels:
        code = S.CODES[lab]
        r = S.simulate(seeds2, code, max(s, 0), k if k >= 0 else S.BIG, **kw)
        outs.append(r)
    outs = np.stack(outs)
    f2, c2 = outs[:, :, 0], outs[:, :, 2]
    nfam = len(cand)
    ib2 = int(np.argmax(f2[:nfam].mean(1)))
    ic2 = int(np.argmax(c2[:nfam].mean(1)))
    iex = cand.index(0)
    for q, (lab, s, k) in enumerate(labels):
        rows.append(dict(pid=pt['pid'], planner=lab, s=s, k=k,
                         fin=float(f2[q].mean()), fin_se=float(f2[q].std(ddof=1) / np.sqrt(n2)),
                         cap=float(c2[q].mean()), cap_se=float(c2[q].std(ddof=1) / np.sqrt(n2)),
                         meanc=float(outs[q, :, 1].mean()), gini=float(outs[q, :, 3].mean()),
                         dens=float(outs[q, :, 4].mean()), nrec=float(outs[q, :, 5].mean()),
                         d_best=float((f2[q] - f2[ib2]).mean()), d_best_se=float((f2[q] - f2[ib2]).std(ddof=1) / np.sqrt(n2)),
                         d_excl=float((f2[q] - f2[iex]).mean()), d_excl_se=float((f2[q] - f2[iex]).std(ddof=1) / np.sqrt(n2)),
                         dcap_best=float((c2[q] - c2[ic2]).mean()), dcap_best_se=float((c2[q] - c2[ic2]).std(ddof=1) / np.sqrt(n2)),
                         is_best=int(q == ib2), is_capbest=int(q == ic2)))
    return stage1, rows


def work(pt):
    t0 = time.time()
    try:
        s1, s2 = evaluate(pt)
        return pt, s1, s2, time.time() - t0, None
    except Exception as e:  # pragma: no cover
        return pt, [], [], time.time() - t0, repr(e)


def cost(pt):
    return (pt.get('n', 16) / 16) ** 2 * pt.get('T', 15)


if __name__ == '__main__':
    outdir, nw = sys.argv[1], int(sys.argv[2])
    quick = len(sys.argv) > 3
    os.makedirs(outdir, exist_ok=True)
    S.simulate(np.arange(4))                       # compile before fork
    pts = grids()
    if quick:
        globals()['N1'], globals()['N2'] = 200, 500
        pts = pts[::97]
    pts_sorted = sorted(pts, key=cost, reverse=True)
    json.dump([{k: v for k, v in p.items()} for p in pts], open(os.path.join(outdir, 'points.json'), 'w'))
    import csv
    f1 = open(os.path.join(outdir, 'stage1.csv'), 'w', newline=''); w1 = None
    f2 = open(os.path.join(outdir, 'stage2.csv'), 'w', newline=''); w2 = None
    t0 = time.time(); done = 0
    ctx = mp.get_context('fork')
    with ctx.Pool(nw) as pool:
        for pt, s1, s2, dt, err in pool.imap_unordered(work, pts_sorted, chunksize=1):
            done += 1
            if err:
                print('ERR', pt['pid'], err, flush=True)
                continue
            if w1 is None:
                w1 = csv.DictWriter(f1, fieldnames=list(s1[0].keys())); w1.writeheader()
                w2 = csv.DictWriter(f2, fieldnames=list(s2[0].keys())); w2.writeheader()
            w1.writerows(s1); w2.writerows(s2)
            if done % 50 == 0 or done == len(pts):
                f1.flush(); f2.flush()
                print(f'{done}/{len(pts)} {time.time()-t0:.0f}s', flush=True)
    f1.close(); f2.close()
    print('total', time.time() - t0)
