"""Original-reward objectives (revision): discounted mean capital (gamma = 0.99) with and without a
penalty for rejected recommendations, as in McKee et al. SI E3 (penalty approximated by single-decider
rejections, an upper bound on 'rejected by both endpoints').  Full family, samples A (select) and B (evaluate).
Usage: python3 ph_reward.py <emp_points.json> <outdir> <nworkers>
"""
import sys, os, json, csv, time
import numpy as np, multiprocessing as mp
import ph_sim as S, ph_sweep as W

N = 10000


def work(pt):
    kw = W.simkw(pt); T = 15
    rules = S.distinct_rules(T)
    rows = []
    for smp, off in (('A', 0), ('B', 1)):
        seeds = np.arange(N, dtype=np.int64) + 60000 * (pt['block'] + off)
        outs = np.stack([S.simulate(seeds, 0, s, k, **kw) for s, k in rules])
        for q, (s, k) in enumerate(rules):
            r = dict(pid=pt['pid'], label=pt.get('label', ''), sample=smp, s=s, k=k)
            for c, nm in ((0, 'fin'), (1, 'meanc'), (2, 'cap'), (6, 'dcap'), (7, 'dutil')):
                x = outs[q, :, c]; r[nm] = float(x.mean())
                dl = x - outs[0, :, c]; r['dE_' + nm] = float(dl.mean()); r['dE_' + nm + '_se'] = float(dl.std(ddof=1) / np.sqrt(N))
            rows.append(r)
    return rows


if __name__ == '__main__':
    emp, outdir, nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    S.simulate(np.arange(4))
    pts = [p for p in W.grids() if p['grid'] == 'kw']
    for p in pts:
        p['block'] = 30000 + 2 * p['pid']
    for q, p in enumerate(json.load(open(emp))):
        p['block'] = 31000 + 2 * q; p['pid'] = 50000 + q; pts.append(p)
    t0 = time.time()
    with mp.get_context('fork').Pool(nw) as pool:
        res = pool.map(work, pts, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(outdir, 'reward.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('done', time.time() - t0)
