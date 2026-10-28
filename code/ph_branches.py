"""Re-run the matched clock-rule branches with additive group contributions (revision audit).

Same points and seeds as ph_cert.py (published-slope points, seed blocks 9000+q) and ph_emp3.py
(empirical points, seed blocks 9500+q), so overall cooperation reproduces traj.csv / traj_emp.csv
exactly.  In addition the per-game contributions of the round-s defectors and cooperators to the
population cooperation rate (cD/n, cC/n) are recorded, which sum exactly to the overall change.

Usage: python ph_branches.py <placement_points.json> <out.csv> <nworkers>
"""
import sys, json, csv
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
import ph_sim as S
import ph_sweep as W

PUB = [dict(label=f'orig_k{k:.1f}', kappa=k) for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)]
PUB += [dict(label=f'orig_k{k:.1f}_nocap', kappa=k, D0=1e6) for k in (0.6, 0.8, 1.0)]


def emp_points(pp):
    d = {p['label']: p for p in pp}
    keep = lambda p: {k: v for k, v in p.items() if k in W.SIMKEYS}
    m1w, m3 = keep(d['pooled_M1w']), keep(d['pooled_M3'])
    k1 = dict(m1w); k1['kappa'] = 1.0
    nc = dict(m1w); nc['D0'] = 1e6
    return [dict(m1w, label='emp_M1w'), dict(k1, label='emp_M1w_k1.0'), dict(nc, label='emp_M1w_nocap'), dict(m3, label='emp_M3')]


def work(args):
    base, q, pt, s = args
    kw = W.simkw({k: v for k, v in pt.items() if k != 'label'})
    seeds = np.arange(20000, dtype=np.int64) + 60000 * (base + q)
    res = [S.simulate_traj(seeds, 0, ss, S.BIG, branch=s, **kw) for ss in (s, s - 1)]
    (o1, t1), (o0, t0) = res
    G = len(seeds)
    rows = []
    for t in range(15):
        rows.append(dict(label=pt['label'], branch=s, round=t + 1, coop_s=t1[t, 0], coop_sm1=t0[t, 0],
                         popD_s=t1[t, 8], popD_sm1=t0[t, 8], popC_s=t1[t, 9], popC_sm1=t0[t, 9],
                         deg_s=t1[t, 3], deg_sm1=t0[t, 3], nD=t1[t, 6]))
    d = o1[:, 0] - o0[:, 0]
    rows[-1]['dfin'] = float(d.mean()); rows[-1]['dfin_se'] = float(d.std(ddof=1) / np.sqrt(G))
    return rows


if __name__ == '__main__':
    pp = json.load(open(sys.argv[1])); out = sys.argv[2]; nw = int(sys.argv[3])
    S.simulate_traj(np.arange(4), 0, 3, S.BIG, branch=3)
    tasks = [(9000, q, p, s) for q, p in enumerate(PUB) for s in range(1, 15)]
    tasks += [(9500, q, p, s) for q, p in enumerate(emp_points(pp)) for s in range(1, 15)]
    with mp.get_context('spawn').Pool(nw) as pool:
        res = pool.map(work, tasks, chunksize=1)
    rows = [r for rr in res for r in rr]
    keys = list(rows[0].keys()) + ['dfin', 'dfin_se']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
    print('rows', len(rows))
