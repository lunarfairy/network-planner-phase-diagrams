"""Empirical placement of the seven human conditions of McKee et al. (2023) on the phase diagram.

1. Rebuild round-by-round networks from the edge-level OSF files; build the decision panel.
2. Estimate acceptance probabilities phi and three random-intercept behavioural models by
   Gauss-Hermite maximum likelihood:
     M1   logit P(C_it) = b_int + b_s deg + k_i (b_n xn + b_r xr) + theta_i,  k_i = 1 (prev C) or kappa
     M1w  P = (1 - w) * M1 + w * P_imit   (Fermi imitation of a random neighbour, tau = 0.1)
     M2   M1 + delta * 1[prev D]           (state-dependence robustness check)
   Round 0: logit = b00 + b01 theta_i, theta_i ~ N(mu, sigma^2), mu fixed at -0.304 (normalisation).
   Fits: pooled, per condition, leave-one-condition-out; group bootstrap (B draws).
3. Strategy optimisation (ph_sweep.evaluate) at the pooled estimate, at each condition's estimate
   and at pooled-bootstrap draws -> phase probabilities.
4. Out-of-sample prediction of the seven human conditions (SI vs pooled vs LOCO vs own parameters).
Usage: python3 ph_empirical.py <osf_zip> <outdir> <nworkers> <B>
"""
import sys, os, time, json, zipfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.special import expit
import ph_sim as S
import ph_sweep as W

MU0 = -0.304
TAU = 0.1
GH_X, GH_W = np.polynomial.hermite.hermgauss(24)
GH_W = GH_W / np.sqrt(np.pi)
IU = np.triu_indices(16, 1)
COND = {'Static network': 'static', 'Random rec.': 'random', 'Coop. clustering': 'clustering',
        'GraphNet planner': 'graphnet', 'Encourag. planner': 'encouragement', 'Neutral planner': 'neutral',
        'Max. connectivity': 'maxconn', 'Max connectivity': 'maxconn'}
SIMCODE = dict(static=1, random=2, clustering=6, graphnet=5, encouragement=5, neutral=4, maxconn=3)
ORDER = ['static', 'random', 'clustering', 'graphnet', 'encouragement', 'neutral', 'maxconn']


# ----------------------------------------------------------------- data
def build_panel(zpath):
    z = zipfile.ZipFile(zpath)
    obs, rec, outc = [], [], []
    for pre in ('baseline', 'evaluation', 'validation'):
        g = pd.read_csv(z.open(f'{pre}_graph_struct_data.csv'))
        c = pd.read_csv(z.open(f'{pre}_cooperation_data.csv'))
        g['cond'] = g.Condition.map(COND); c['cond'] = c.Condition.map(COND)
        g = g.sort_values(['Group', 'Round', 'Edge'])
        nxt = g.groupby(['Group', 'Edge']).Prev_Status.shift(-1)
        g['acc_inf'] = np.where(g.Recommendation == 1, nxt == 1, nxt == 0).astype(float)
        g.loc[nxt.isna(), 'acc_inf'] = np.nan
        if 'Referent_Player_Coop' not in g.columns:
            g['Referent_Player_Coop'] = np.nan
            g['Accepted'] = np.nan
        r = g[(g.Recommendation != 0) & g.acc_inf.notna()].copy()
        rec.append(r[['Group', 'cond', 'Round', 'Recommendation', 'Referent_Player_Coop', 'Accepted', 'acc_inf', 'Edge_Type']])
        for grp, gc in c.groupby('Group'):
            cond = gc.cond.iloc[0]
            Y = np.full((15, 16), np.nan); PAY = np.full((15, 16), np.nan); CAP = np.full((15, 16), np.nan)
            Y[gc.Round.values, gc.Player.values] = gc.Cooperation.values
            PAY[gc.Round.values, gc.Player.values] = gc.Payoff.values
            CAP[gc.Round.values, gc.Player.values] = gc.Capital.values
            gg = g[g.Group == grp]
            As = np.zeros((15, 16, 16))
            for rr, d in gg.groupby('Round'):
                M = np.zeros((16, 16)); M[IU[0][d.Edge.values], IU[1][d.Edge.values]] = d.Prev_Status.values
                As[rr] = M + M.T
            for rr in range(15):
                outc.append(dict(group=grp, cond=cond, round=rr, coop=np.nanmean(Y[rr])))
                for i in range(16):
                    if np.isnan(Y[rr, i]):
                        continue
                    A = As[rr]
                    nb = np.flatnonzero(A[i] > 0)
                    deg = len(nb)
                    row = dict(group=grp, cond=cond, pid=f'{grp}_{i}', round=rr, y=Y[rr, i], deg=deg)
                    if rr == 0:
                        row.update(ap=np.nan, xn=np.nan, xr=np.nan, mimit=np.nan, bind=0)
                    else:
                        ap = Y[rr - 1, i]
                        if np.isnan(ap):
                            continue
                        yn = Y[rr - 1, nb]
                        xn = float(np.nansum(yn))
                        xr = xn / deg if deg > 0 else 0.0
                        ok = ~np.isnan(yn)
                        mt = {}
                        for tau_ in (0.03, 0.1, 0.3):
                            if ok.sum() > 0 and not np.isnan(PAY[rr - 1, i]):
                                F = expit((PAY[rr - 1, nb[ok]] - PAY[rr - 1, i]) / tau_)
                                mt[tau_] = float(np.mean(F * yn[ok] + (1 - F) * ap))
                            else:
                                mt[tau_] = np.nan
                        mimit = mt[0.1]
                        row.update(mimit_t003=mt[0.03], mimit_t03=mt[0.3])
                        capprev = CAP[rr - 1, i]
                        bind = int((not np.isnan(capprev)) and 0.05 * deg > capprev + 1e-9)
                        row.update(ap=ap, xn=xn, xr=xr, mimit=mimit, bind=bind)
                    obs.append(row)
    return pd.DataFrame(obs), pd.concat(rec), pd.DataFrame(outc)


# ----------------------------------------------------------------- likelihood
class Data:
    def __init__(self, df):
        df = df[df.bind == 0].copy()
        codes, uniq = pd.factorize(df.pid)
        self.np_ = len(uniq)
        self.pidx = codes
        self.r0 = (df['round'].values == 0)
        self.y = df.y.values
        self.deg = df.deg.values.astype(float)
        self.xn = np.nan_to_num(df.xn.values)
        self.xr = np.nan_to_num(df.xr.values)
        self.prevD = (np.nan_to_num(df.ap.values, nan=1.0) == 0)
        m = df.mimit.values
        self.hasm = ~np.isnan(m) & ~self.r0
        self.m = np.nan_to_num(m)
        self.n = len(df)


NAMES = {'M1': ['b00', 'b01', 'b_int', 'b_s', 'b_n', 'b_r', 'log_sigma', 'kappa'],
         'M1w': ['b00', 'b01', 'b_int', 'b_s', 'b_n', 'b_r', 'log_sigma', 'kappa', 'logit_w'],
         'M2': ['b00', 'b01', 'b_int', 'b_s', 'b_n', 'b_r', 'log_sigma', 'kappa', 'delta']}
X0 = {'M1': [1.807, 0.818, -0.010, -0.193, 0.370, 1.521, np.log(2.41), 1.0]}
X0['M1w'] = X0['M1'] + [-2.0]
X0['M2'] = X0['M1'] + [0.0]


def negll(x, D, model):
    b00, b01, bi, bs, bn, br, ls, kap = x[:8]
    sig = np.exp(ls)
    th = MU0 + np.sqrt(2) * sig * GH_X                       # (Q,)
    k = np.where(D.prevD, kap, 1.0)
    eta = bi + bs * D.deg + k * (bn * D.xn + br * D.xr)
    if model == 'M2':
        eta = eta + x[8] * D.prevD
    lin = np.where(D.r0[:, None], b00 + b01 * th[None, :], eta[:, None] + th[None, :])
    p = expit(lin)
    if model == 'M1w':
        w = expit(x[8])
        p = np.where(D.hasm[:, None], (1 - w) * p + w * D.m[:, None], p)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = np.where(D.y[:, None] == 1, np.log(p), np.log1p(-p))     # (N, Q)
    lp = np.zeros((D.np_, len(th)))
    np.add.at(lp, D.pidx, ll)
    mx = lp.max(1, keepdims=True)
    return -np.sum(mx[:, 0] + np.log(np.exp(lp - mx) @ GH_W))


def fit(df, model, x0=None):
    D = Data(df)
    x0 = np.array(X0[model] if x0 is None else x0, float)
    r = minimize(negll, x0, args=(D, model), method='L-BFGS-B')
    out = dict(zip(NAMES[model], r.x))
    out.update(nll=r.fun, n_obs=D.n, n_players=D.np_, converged=bool(r.success), model=model)
    out['sigma'] = float(np.exp(out['log_sigma']))
    if model == 'M1w':
        out['w'] = float(expit(out['logit_w']))
    return out


PHI_KEYS = ('phi_del_refD', 'phi_del_refC', 'phi_add_refD', 'phi_add_refC')


def phi_est(rec):
    """ML acceptance probabilities phi[del|ref D, del|ref C, add|ref D, add|ref C].
    Referent known (baseline files, or C-C / D-D pairs): Bernoulli(phi[type]).  Referent unknown
    on a C-D pair (evaluation/validation files): the deciding endpoint is random, so the acceptance
    probability is the average of the two referent types."""
    out = {}
    if len(rec) == 0:
        for k in PHI_KEYS:
            out[k] = np.nan; out['n_' + k] = 0
        return out
    v = rec.Recommendation.values
    y = rec.acc_inf.values
    et = rec.Edge_Type.values
    ref = rec.Referent_Player_Coop.values.astype(float)
    ref = np.where(np.isnan(ref) & (et == 'C-C'), 1.0, ref)
    ref = np.where(np.isnan(ref) & (et == 'D-D'), 0.0, ref)
    base = np.where(v == -1, 0, 2)
    known = ~np.isnan(ref)
    idx_known = base + np.nan_to_num(ref).astype(int)          # refD -> +0, refC -> +1

    def nll(z):
        ph = np.clip(expit(z), 1e-9, 1 - 1e-9)
        p = np.where(known, ph[idx_known], 0.5 * (ph[base] + ph[base + 1]))
        return -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
    r = minimize(nll, np.zeros(4), method='L-BFGS-B')
    ph = expit(r.x)
    for q, k in enumerate(PHI_KEYS):
        m = (base == (0 if q < 2 else 2))
        nk = int((m & known & (idx_known == q)).sum())
        out[k] = float(ph[q]) if nk > 0 or (m & ~known).sum() > 0 else np.nan
        out['n_' + k] = nk
    return out


def resample_groups(df, rec, rng):
    parts, rparts = [], []
    for cond, dc in df.groupby('cond'):
        gs = dc.group.unique()
        pick = rng.choice(gs, len(gs), replace=True)
        for q, gsel in enumerate(pick):
            d = dc[dc.group == gsel].copy(); d['pid'] = d.pid + f'_b{q}'
            parts.append(d)
            rparts.append(rec[rec.Group == gsel])
    return pd.concat(parts), pd.concat(rparts)


def boot_task(args):
    b, subset = args
    rng = np.random.default_rng(1000 + b)
    df = PANEL if subset == 'pooled' else PANEL[PANEL.cond == subset]
    rc = REC if subset == 'pooled' else REC[REC.cond == subset]
    dfb, rcb = resample_groups(df, rc, rng)
    res = dict(b=b, subset=subset)
    for m in ('M1', 'M1w', 'M2'):
        f = fit(dfb, m)
        for kk in ('kappa', 'sigma', 'b_int', 'b_s', 'b_n', 'b_r', 'b00', 'b01', 'converged'):
            res[f'{m}_{kk}'] = f[kk]
        if m == 'M1w':
            res['M1w_w'] = f['w']
        if m == 'M2':
            res['M2_delta'] = f['delta']
    res.update(phi_est(rcb))
    return res


def point_task(args):
    name, sub_df, sub_rec = args
    out = []
    for m in ('M1', 'M1w', 'M2'):
        f = fit(sub_df, m); f['subset'] = name; out.append(f)
    ph = phi_est(sub_rec); ph['subset'] = name
    return out, ph


def params_to_pt(fw, ph, pid, extra=None):
    phi = S.PHI0.copy()
    if ph is not None:
        for q, key in enumerate(PHI_KEYS):
            if ph.get(key) is not None and not np.isnan(ph[key]):
                phi[q] = ph[key]
    pt = dict(grid='empirical', pid=pid, kappa=float(fw['kappa']), w=float(fw.get('w', 0.0)),
              sd_th=float(fw['sigma']), mu_th=MU0, phi=phi.tolist(),
              betas=[fw['b00'], fw['b01'], fw['b_int'], fw['b_s'], fw['b_n'], fw['b_r']])
    if extra:
        pt.update(extra)
    return pt


def opt_task(pt):
    s1, s2 = W.evaluate(pt)
    for r in s2:
        r['label'] = pt['label']
    return s2


def oos_task(args):
    label, cond, pt = args
    kw = W.simkw(pt)
    seeds = np.arange(10000, dtype=np.int64) + 60000 * (60000 + ORDER.index(cond))
    r = S.simulate(seeds, SIMCODE[cond], 0, S.BIG, **kw)
    return dict(paramset=label, cond=cond, sim_final=float(r[:, 0].mean()), sim_final_se=float(r[:, 0].std(ddof=1) / 100),
                sim_mean=float(r[:, 1].mean()))


if __name__ == '__main__':
    zpath, outdir, nw, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    PANEL, REC, OUTC = build_panel(zpath)
    PANEL.to_csv(os.path.join(outdir, 'panel.csv.gz'), index=False)
    print('panel', PANEL.shape, 'bind', PANEL.bind.sum(), time.time() - t0, flush=True)
    S.simulate(np.arange(4))
    ctx = mp.get_context('fork')
    # ---- point estimates: pooled, per condition, LOCO
    jobs = [('pooled', PANEL, REC)]
    for c in ORDER:
        jobs.append((c, PANEL[PANEL.cond == c], REC[REC.cond == c]))
        jobs.append(('loco_' + c, PANEL[PANEL.cond != c], REC[REC.cond != c]))
    with ctx.Pool(min(nw, len(jobs))) as pool:
        res = pool.map(point_task, jobs)
    fits = pd.DataFrame([f for r in res for f in r[0]])
    phis = pd.DataFrame([r[1] for r in res])
    for col, tau_ in (('mimit_t003', 0.03), ('mimit_t03', 0.3)):
        dft = PANEL.copy(); dft['mimit'] = dft[col]
        f = fit(dft, 'M1w'); f['subset'] = f'pooled_tau{tau_}'
        fits = pd.concat([fits, pd.DataFrame([f])], ignore_index=True)
    fits.to_csv(os.path.join(outdir, 'fits.csv'), index=False)
    phis.to_csv(os.path.join(outdir, 'phi.csv'), index=False)
    print('point fits', time.time() - t0, flush=True)
    print(fits[fits.model == 'M1w'][['subset', 'kappa', 'w', 'sigma', 'b_n', 'b_r', 'nll', 'converged']].to_string(), flush=True)
    # ---- bootstrap
    tasks = [(b, s) for s in ['pooled'] + ORDER for b in range(B)]
    with ctx.Pool(nw) as pool:
        boots = pool.map(boot_task, tasks, chunksize=1)
    boots = pd.DataFrame(boots)
    boots.to_csv(os.path.join(outdir, 'bootstrap.csv'), index=False)
    print('bootstrap', time.time() - t0, flush=True)
    # ---- strategy optimisation at empirical parameters
    fw = fits[fits.model == 'M1w'].set_index('subset')
    ph = phis.set_index('subset')
    pts = []
    pid = 20000
    for sub in ['pooled'] + ORDER:
        pts.append(params_to_pt(fw.loc[sub], ph.loc[sub].to_dict(), pid, dict(label=sub))); pid += 1
    bp = boots[boots.subset == 'pooled']
    for _, r in bp.iterrows():
        f = dict(kappa=r.M1w_kappa, w=r.M1w_w, sigma=r.M1w_sigma, b00=r.M1w_b00, b01=r.M1w_b01, b_int=r.M1w_b_int,
                 b_s=r.M1w_b_s, b_n=r.M1w_b_n, b_r=r.M1w_b_r)
        pts.append(params_to_pt(f, r.to_dict(), pid, dict(label=f'boot_{int(r.b)}'))); pid += 1
    # kappa slice through the pooled empirical population (all other parameters at pooled M1w)
    for kap in np.linspace(0, 1.5, 16):
        f = dict(fw.loc['pooled']); f['kappa'] = float(kap)
        pts.append(params_to_pt(f, ph.loc['pooled'].to_dict(), pid, dict(label=f'kslice_{kap:.2f}'))); pid += 1
    # SI parameters, and pooled M1 (no imitation) for comparison
    pts.append(dict(grid='empirical', pid=pid, label='SI', kappa=1.0, w=0.0)); pid += 1
    f1 = fits[(fits.model == 'M1') & (fits.subset == 'pooled')].iloc[0]
    pts.append(params_to_pt(dict(f1), ph.loc['pooled'].to_dict(), pid, dict(label='pooled_M1', w=0.0))); pid += 1
    with ctx.Pool(nw) as pool:
        opt = pool.map(opt_task, pts, chunksize=1)
    pd.DataFrame([r for rr in opt for r in rr]).to_csv(os.path.join(outdir, 'empirical_opt.csv'), index=False)
    json.dump(pts, open(os.path.join(outdir, 'empirical_points.json'), 'w'))
    print('optimisation', time.time() - t0, flush=True)
    # ---- out-of-sample prediction of the seven human conditions
    oos = []
    si = dict(grid='x', pid=0, kappa=1.0, w=0.0)
    f1 = fits[fits.model == 'M1'].set_index('subset')
    for c in ORDER:
        oos.append(('SI', c, si))
        oos.append(('pooled_M1w', c, params_to_pt(fw.loc['pooled'], ph.loc['pooled'].to_dict(), 0)))
        oos.append(('loco_M1w', c, params_to_pt(fw.loc['loco_' + c], ph.loc['loco_' + c].to_dict(), 0)))
        oos.append(('own_M1w', c, params_to_pt(fw.loc[c], ph.loc[c].to_dict() if c != 'static' else None, 0)))
        oos.append(('pooled_M1', c, params_to_pt(dict(f1.loc['pooled']), ph.loc['pooled'].to_dict(), 0, dict(w=0.0))))
        oos.append(('loco_M1', c, params_to_pt(dict(f1.loc['loco_' + c]), ph.loc['loco_' + c].to_dict(), 0, dict(w=0.0))))
    with ctx.Pool(min(nw, len(oos))) as pool:
        o = pool.map(oos_task, oos)
    o = pd.DataFrame(o)
    fin = OUTC[OUTC['round'] == 14].groupby(['cond', 'group']).coop.mean().groupby('cond').agg(['mean', 'sem', 'count'])
    mean_all = OUTC.groupby(['cond', 'group']).coop.mean().groupby('cond').mean()
    o['obs_final'] = o.cond.map(fin['mean']); o['obs_final_se'] = o.cond.map(fin['sem']); o['n_groups'] = o.cond.map(fin['count'])
    o['obs_mean'] = o.cond.map(mean_all)
    o.to_csv(os.path.join(outdir, 'oos.csv'), index=False)
    OUTC.to_csv(os.path.join(outdir, 'observed_rounds.csv'), index=False)
    print('done', time.time() - t0, flush=True)
