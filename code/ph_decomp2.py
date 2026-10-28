"""Finer parameter-replacement decomposition of the response coefficients (revision audit).

The original decomposition (ph_emp3.py) exchanged the later-round coefficients (beta_0, beta_s, beta_n,
beta_r) as one block between the published simulator and the re-estimated M1w population.  Here beta_r
and each of the other coefficients are exchanged separately.  Each point is evaluated like a grid point
(all 50 rules + 2 references, samples A and B of 10,000 games; seed blocks 33000+).

Usage: python ph_decomp2.py <placement_points.json> <out.csv> <nworkers>
"""
import sys, json, csv
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
import ph_sim as S
import ph_cert as C

SI_B = [1.807, 0.818, -0.010, -0.193, 0.370, 1.521]
IDX = dict(b_int=2, b_s=3, b_n=4, b_r=5)


def points(pp):
    emp = {p['label']: p for p in pp}['pooled_M1w']
    EB = list(emp['betas']); EPHI = emp['phi']
    base_si = dict(kappa=1.0, w=0.0, sd_th=2.41, mu_th=-0.304, phi=list(S.PHI0), betas=SI_B)
    base_emp = dict(kappa=emp['kappa'], w=0.0, sd_th=emp['sd_th'], mu_th=-0.304, phi=EPHI, betas=EB)
    sets = {'b_r': ['b_r'], 'b_int': ['b_int'], 'b_s': ['b_s'], 'b_n': ['b_n'], 'b_int_s_n': ['b_int', 'b_s', 'b_n']}
    pts = []
    for nm, keys in sets.items():
        b = list(SI_B)
        for k in keys: b[IDX[k]] = EB[IDX[k]]
        pts.append(dict(base_si, betas=b, label=f'decomp2_SI_plus_{nm}'))
        b = list(EB)
        for k in keys: b[IDX[k]] = SI_B[IDX[k]]
        pts.append(dict(base_emp, betas=b, label=f'decomp2_EMP_minus_{nm}'))
    for q, p in enumerate(pts):
        p['pid'] = 60000 + q; p['block'] = 33000 + 2 * q; p['grid'] = 'decomp2'
    return pts


def work(args):
    pt, smp = args
    rows = C.evaluate_full(pt, 10000, pt['block'] + (0 if smp == 'A' else 1))
    for r in rows:
        r['sample'] = smp; r['label'] = pt['label']
    return rows


if __name__ == '__main__':
    pp = json.load(open(sys.argv[1])); out = sys.argv[2]; nw = int(sys.argv[3])
    pts = points(pp)
    json.dump(pts, open(out.replace('.csv', '_points.json'), 'w'))
    tasks = [(p, s) for p in pts for s in ('A', 'B')]
    with mp.get_context('spawn').Pool(nw) as pool:
        res = pool.map(work, tasks, chunksize=1)
    rows = [r for rr in res for r in rr]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('rows', len(rows))
