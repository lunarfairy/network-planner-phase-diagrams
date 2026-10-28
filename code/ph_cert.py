"""Certified re-evaluation of the strategy phase diagram (revision; run on h20).

For every grid point of ph_sweep.grids() plus new compliance grids, all 50 behaviourally distinct
rules of Pi(s, k) (ph_sim.distinct_rules) and the encouragement / static reference planners are
evaluated on TWO fresh, independent samples of N games each (A and B; same-seed coupling of all
rules within a sample).  Sample A is used to select (phase classification, winners for the three
objectives); sample B, never used for selection, evaluates the locked selections and gives an
independent replicate classification.  No candidate screening, no cap.

A second part records per-round trajectories of matched clock rules (s vs s-1, which are identical
up to the planner step after round s) for the mechanism analysis, with and without the capital
constraint.

Usage: python3 ph_cert.py <outdir> <nworkers> [quick]
"""
import sys, os, time, json, itertools, csv
import numpy as np
import multiprocessing as mp
import ph_sim as S
import ph_sweep as W

N = 10000
KAPPAS = W.KAPPAS


def all_points():
    pts = W.grids()                                 # pids 0..1223, identical to the original sweep
    pid = len(pts)
    for lam in (0.1, 0.25, 0.5, 0.75):
        for kap in KAPPAS:
            pts.append(dict(grid='comp_all', kappa=kap, phi=(S.PHI0 * lam).tolist(), lam=lam, pid=pid)); pid += 1
    for lam in (0.25, 0.5):
        for kap in KAPPAS:
            ph = S.PHI0.copy(); ph[2:] *= lam
            pts.append(dict(grid='comp_add', kappa=kap, phi=ph.tolist(), lam=lam, pid=pid)); pid += 1
            ph = S.PHI0.copy(); ph[:2] *= lam
            pts.append(dict(grid='comp_del', kappa=kap, phi=ph.tolist(), lam=lam, pid=pid)); pid += 1
    return pts


def evaluate_full(pt, n, block):
    kw = W.simkw(pt)
    T = int(pt.get('T', 15))
    rules = S.distinct_rules(T)
    seeds = np.arange(n, dtype=np.int64) + 60000 * block
    assert seeds.max() < 2 ** 32
    labs = [('family', s, k) for s, k in rules] + [('encouragement', -1, -1), ('static', -1, -1)]
    outs = np.stack([S.simulate(seeds, S.CODES[l], max(s, 0), k if k >= 0 else S.BIG, **kw) for l, s, k in labs])
    nf = len(rules)
    iE = 0
    iC = rules.index((T - 1, S.BIG))
    rows = []
    best = {}
    for q, name in ((0, 'fin'), (1, 'meanc'), (2, 'cap')):
        best[name] = int(np.argmax(outs[:nf, :, q].mean(1)))
    for r_, (l, s, k) in enumerate(labs):
        row = dict(pid=pt['pid'], planner=l, s=s, k=k)
        for q, name in ((0, 'fin'), (1, 'meanc'), (2, 'cap')):
            x = outs[r_, :, q]
            row[name] = float(x.mean()); row[name + '_se'] = float(x.std(ddof=1) / np.sqrt(n))
            for ref, iref in (('E', iE), ('C', iC), ('best', best[name])):
                dlt = x - outs[iref, :, q]
                row[f'd{ref}_{name}'] = float(dlt.mean())
                row[f'd{ref}_{name}_se'] = float(dlt.std(ddof=1) / np.sqrt(n))
        row['gini'] = float(outs[r_, :, 3].mean()); row['dens'] = float(outs[r_, :, 4].mean())
        row['nrec'] = float(outs[r_, :, 5].mean())
        rows.append(row)
    # variance reduction achieved by same-seed coupling (fin, E vs C)
    vr = float(np.var(outs[iE, :, 0] - outs[iC, :, 0]) / (np.var(outs[iE, :, 0]) + np.var(outs[iC, :, 0]) + 1e-12))
    for row in rows:
        row['crn_var_ratio_EC'] = vr
    return rows


def work(args):
    pt, sample = args
    t0 = time.time()
    try:
        block = 5000 + 2 * pt['pid'] + (0 if sample == 'A' else 1)
        rows = evaluate_full(pt, N, block)
        for r in rows:
            r['sample'] = sample
        return rows, time.time() - t0, None
    except Exception as e:  # pragma: no cover
        return [], time.time() - t0, f"{pt['pid']} {sample} {e!r}"


TRAJ_PTS = [dict(label=f'orig_k{k:.1f}', kappa=k) for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)]
TRAJ_PTS += [dict(label=f'orig_k{k:.1f}_nocap', kappa=k, D0=1e6) for k in (0.6, 0.8, 1.0)]


def traj_work(args):
    q, pt, s = args
    kw = {kk: v for kk, v in pt.items() if kk in W.SIMKEYS}
    seeds = np.arange(20000, dtype=np.int64) + 60000 * (9000 + q)
    res = []
    for ss in (s, s - 1):
        out, tr = S.simulate_traj(seeds, 0, ss, S.BIG, branch=s, **kw)
        res.append((out, tr))
    (o1, t1), (o0, t0) = res
    rows = []
    for t in range(15):
        rows.append(dict(label=pt['label'], branch=s, round=t + 1,
                         coop_s=t1[t, 0], coop_sm1=t0[t, 0], coopD_s=t1[t, 1], coopD_sm1=t0[t, 1],
                         coopC_s=t1[t, 2], coopC_sm1=t0[t, 2], deg_s=t1[t, 3], deg_sm1=t0[t, 3],
                         ncd_s=t1[t, 4], ncd_sm1=t0[t, 4], nbind_s=t1[t, 5], nbind_sm1=t0[t, 5],
                         nD=t1[t, 6]))
    d = o1[:, 0] - o0[:, 0]
    rows[-1]['dfin'] = float(d.mean()); rows[-1]['dfin_se'] = float(d.std(ddof=1) / np.sqrt(len(d)))
    return rows


def cost(pt):
    return (pt.get('n', 16) / 16) ** 2 * pt.get('T', 15)


if __name__ == '__main__':
    outdir, nw = sys.argv[1], int(sys.argv[2])
    quick = len(sys.argv) > 3
    os.makedirs(outdir, exist_ok=True)
    S.simulate(np.arange(4)); S.simulate_traj(np.arange(4), 0, 3, S.BIG, branch=3)
    pts = all_points()
    if quick:
        N = 300
        pts = pts[::150]
    json.dump(pts, open(os.path.join(outdir, 'cert_points.json'), 'w'))
    tasks = sorted([(p, smp) for p in pts for smp in ('A', 'B')], key=lambda a: -cost(a[0]))
    ctx = mp.get_context('fork')
    t0 = time.time()
    # ---- trajectories (small) first
    ttasks = [(q, p, s) for q, p in enumerate(TRAJ_PTS) for s in range(1, 15)]
    if quick:
        ttasks = ttasks[:3]
    with ctx.Pool(nw) as pool:
        tr = pool.map(traj_work, ttasks, chunksize=1)
    with open(os.path.join(outdir, 'traj.csv'), 'w', newline='') as f:
        rows = [r for rr in tr for r in rr]
        keys = list(rows[0].keys()) + ['dfin', 'dfin_se']
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
    print('traj done', time.time() - t0, flush=True)
    f = open(os.path.join(outdir, 'cert.csv'), 'w', newline=''); wr = None; done = 0
    with ctx.Pool(nw) as pool:
        for rows, dt, err in pool.imap_unordered(work, tasks, chunksize=1):
            done += 1
            if err:
                print('ERR', err, flush=True); continue
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
            wr.writerows(rows)
            if done % 100 == 0 or done == len(tasks):
                f.flush(); print(f'{done}/{len(tasks)} {time.time()-t0:.0f}s', flush=True)
    f.close()
    print('total', time.time() - t0, flush=True)
