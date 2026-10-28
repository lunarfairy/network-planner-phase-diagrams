"""Dense kappa grid near the exclusion / switch / conciliation boundaries (revision, run on h20).

kappa = 0.50, 0.52, ..., 1.20 at w = 0 and w = 0.1 (72 points).  Every point is evaluated exactly as in
ph_cert.py: all 50 distinct rules plus the two references on two independent samples (A, B) of 10,000
games.  Point ids start at 20000, so seed blocks (5000 + 2*pid + sample) do not overlap any other run.

Usage: python3 ph_dense.py <outdir> <nworkers>
"""
import sys, os, time, json, csv
import numpy as np
import multiprocessing as mp
import ph_sim as S
import ph_cert as Cc


def dense_points():
    pts, pid = [], 20000
    for w in (0.0, 0.1):
        for kap in np.round(np.arange(0.50, 1.2001, 0.02), 2):
            pts.append(dict(grid='dense', kappa=float(kap), w=w, pid=pid)); pid += 1
    return pts


if __name__ == '__main__':
    outdir, nw = sys.argv[1], int(sys.argv[2])
    os.makedirs(outdir, exist_ok=True)
    S.simulate(np.arange(4))
    pts = dense_points()
    assert max(5000 + 2 * p['pid'] + 1 for p in pts) * 60000 + Cc.N < 2 ** 32
    json.dump(pts, open(os.path.join(outdir, 'dense_points.json'), 'w'))
    tasks = [(p, smp) for p in pts for smp in ('A', 'B')]
    t0 = time.time(); f = open(os.path.join(outdir, 'dense.csv'), 'w', newline=''); wr = None; done = 0
    with mp.get_context('fork').Pool(nw) as pool:
        for rows, dt, err in pool.imap_unordered(Cc.work, tasks, chunksize=1):
            done += 1
            if err:
                print('ERR', err, flush=True); continue
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
            wr.writerows(rows)
            if done % 24 == 0 or done == len(tasks):
                f.flush(); print(f'{done}/{len(tasks)} {time.time()-t0:.0f}s', flush=True)
    f.close(); print('total', time.time() - t0, flush=True)
