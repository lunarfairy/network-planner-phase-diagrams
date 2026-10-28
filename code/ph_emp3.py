"""Revised empirical analysis (revision round 1; run on h20).

Changes relative to ph_empirical.py / ph_empirical2.py
  * Neighbour covariates use the actions of ALL 16 nodes, decoded from the edge-type log.  Rows of
    dropped-out, bot-controlled nodes are missing from the cooperation files, and the original panel
    coded those neighbours as defectors (variant 'V0', kept for sensitivity).
  * Random-intercept likelihood with analytic gradients, Gauss-Hermite order Q chosen by an explicit
    convergence study (Q in 24..480), multi-start L-BFGS-B.
  * Imitation share w parameterised directly on [0, 0.5] (w = 0 attainable); profile likelihood.
  * Correctly nested heterogeneity tests: M1 (common intercept, common kappa) < M1i (condition
    intercepts, common kappa) < M1c (condition intercepts and condition kappa).
  * Group bootstrap (within condition) of all models and of phi; placement under every bootstrap
    draw with the full family of 50 distinct rules; acceptance probabilities resampled for M3 too.
  * Placement with selection (sample A) and locked evaluation (sample B); condition-specific
    placement from complete M1c parameter vectors; parameter-replacement decomposition; matched
    clock trajectories at the empirical parameters; LOCO prediction intervals.
Usage: python3 ph_emp3.py <osf_zip> <outdir> <nworkers> <B> [quick]
"""
import sys, os, time, json, zipfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.special import expit
from scipy import sparse
import ph_sim as S
import ph_sweep as W
import ph_cert as C
import ph_empirical as E
from archived_history import load_historical_comparison

MU0 = -0.304
ORDER = E.ORDER
IU = np.triu_indices(16, 1)
QDEF = 160
BASELINE = ('static', 'random', 'clustering')


# ============================================================ data
def decode_actions(gr):
    """All 16 actions of one group-round from the unordered edge types of all 120 pairs."""
    a = np.full(16, np.nan)
    e = gr.Edge.values; t = gr.Edge_Type.values
    i, j = IU[0][e], IU[1][e]
    for code, val in (('C-C', 1.0), ('D-D', 0.0)):
        m = t == code
        a[i[m]] = val; a[j[m]] = val
    m = t == 'C-D'
    for _ in range(4):
        if not np.isnan(a).any():
            break
        for ii, jj in zip(i[m], j[m]):
            if np.isnan(a[ii]) and not np.isnan(a[jj]): a[ii] = 1 - a[jj]
            if np.isnan(a[jj]) and not np.isnan(a[ii]): a[jj] = 1 - a[ii]
    return a


def build_panel2(zpath):
    z = zipfile.ZipFile(zpath)
    obs, rec, outc, flow = [], [], [], []
    nmis = 0
    for pre in ('baseline', 'evaluation', 'validation'):
        g = pd.read_csv(z.open(f'{pre}_graph_struct_data.csv'))
        c = pd.read_csv(z.open(f'{pre}_cooperation_data.csv'))
        g['cond'] = g.Condition.map(E.COND); c['cond'] = c.Condition.map(E.COND)
        g = g.sort_values(['Group', 'Round', 'Edge'])
        nxt = g.groupby(['Group', 'Edge']).Prev_Status.shift(-1)
        g['acc_inf'] = np.where(g.Recommendation == 1, nxt == 1, nxt == 0).astype(float)
        g.loc[nxt.isna(), 'acc_inf'] = np.nan
        if 'Referent_Player_Coop' not in g.columns:
            g['Referent_Player_Coop'] = np.nan; g['Accepted'] = np.nan
        r = g[(g.Recommendation != 0) & g.acc_inf.notna()].copy()
        rec.append(r[['Group', 'cond', 'Round', 'Recommendation', 'Referent_Player_Coop', 'Accepted', 'acc_inf', 'Edge_Type']])
        for grp, gg in g.groupby('Group'):
            gc = c[c.Group == grp]
            cond = gg.cond.iloc[0]
            Y = np.full((15, 16), np.nan); PAY = np.full((15, 16), np.nan); CAP = np.full((15, 16), np.nan)
            Y[gc.Round.values, gc.Player.values] = gc.Cooperation.values
            PAY[gc.Round.values, gc.Player.values] = gc.Payoff.values
            CAP[gc.Round.values, gc.Player.values] = gc.Capital.values
            A_all = np.zeros((15, 16)); As = np.zeros((15, 16, 16))
            for rr, d in gg.groupby('Round'):
                A_all[rr] = decode_actions(d)
                M = np.zeros((16, 16)); M[IU[0][d.Edge.values], IU[1][d.Edge.values]] = d.Prev_Status.values
                As[rr] = M + M.T
            human = ~np.isnan(Y)
            nmis += int(np.sum(human & (A_all != Y)))
            # payoffs of every node (donation game, b = 0.10, c = 0.05 per tie)
            PAYc = np.zeros((15, 16))
            for rr in range(15):
                deg = As[rr].sum(1); xn = As[rr] @ A_all[rr]
                PAYc[rr] = 0.10 * xn - 0.05 * deg * A_all[rr]
            dropgrp = int((~human).any())
            for rr in range(15):
                outc.append(dict(group=grp, cond=cond, round=rr, coop=np.nanmean(Y[rr]), coop_all16=A_all[rr].mean(),
                                 n_human=int(human[rr].sum())))
                for i in range(16):
                    fl = dict(group=grp, cond=cond, round=rr, human=int(human[rr, i]))
                    if not human[rr, i]:
                        flow.append(fl); continue
                    A = As[rr]; nb = np.flatnonzero(A[i] > 0); deg = len(nb)
                    row = dict(group=grp, cond=cond, pid=f'{grp}_{i}', round=rr, y=Y[rr, i], deg=deg, dropgrp=dropgrp)
                    if rr == 0:
                        row.update(ap=np.nan, xn=np.nan, xr=np.nan, xn0=np.nan, xr0=np.nan, mimit=np.nan,
                                   mimit_t003=np.nan, mimit_t03=np.nan, bind=0, prev_bot=0, nb_bot=0)
                    else:
                        ap = A_all[rr - 1, i]
                        yn = A_all[rr - 1, nb]
                        xn = float(yn.sum()); xr = xn / deg if deg > 0 else 0.0
                        yn0 = Y[rr - 1, nb]; xn0 = float(np.nansum(yn0)); xr0 = xn0 / deg if deg > 0 else 0.0
                        pay_nb = PAYc[rr - 1, nb]; pay_i = PAYc[rr - 1, i]
                        mt = {}
                        for tau_ in (0.03, 0.1, 0.3):
                            if deg > 0:
                                F = expit((pay_nb - pay_i) / tau_)
                                mt[tau_] = float(np.mean(F * yn + (1 - F) * ap))
                            else:
                                mt[tau_] = np.nan
                        capprev = CAP[rr - 1, i]
                        bind = int((not np.isnan(capprev)) and 0.05 * deg > capprev + 1e-9)
                        row.update(ap=ap, xn=xn, xr=xr, xn0=xn0, xr0=xr0, mimit=mt[0.1], mimit_t003=mt[0.03],
                                   mimit_t03=mt[0.3], bind=bind, prev_bot=int(not human[rr - 1, i]),
                                   nb_bot=int(np.isnan(yn0).sum()))
                    fl['bind'] = row['bind']
                    flow.append(fl)
                    obs.append(row)
    print('action decoding mismatches with recorded human choices:', nmis, flush=True)
    return pd.DataFrame(obs), pd.concat(rec), pd.DataFrame(outc), pd.DataFrame(flow)


# ============================================================ likelihood
class Data:
    def __init__(self, df, mcol='mimit', coding='V1'):
        df = df[df.bind == 0]
        codes, uniq = pd.factorize(df.pid.astype(str))
        self.np_ = len(uniq); self.pidx = codes; self.n = len(df)
        self.P = sparse.csr_matrix((np.ones(self.n), (codes, np.arange(self.n))), shape=(self.np_, self.n))
        self.r0 = df['round'].values == 0
        self.y = df.y.values.astype(float)
        self.deg = df.deg.values.astype(float)
        xn = df.xn if coding == 'V1' else df.xn0
        xr = df.xr if coding == 'V1' else df.xr0
        self.xn = np.nan_to_num(xn.values.astype(float)); self.xr = np.nan_to_num(xr.values.astype(float))
        self.prevD = (np.nan_to_num(df.ap.values.astype(float), nan=1.0) == 0) & ~self.r0
        m = df[mcol].values.astype(float)
        self.hasm = (~np.isnan(m)) & ~self.r0
        self.m = np.nan_to_num(m)
        self.c = pd.Categorical(df.cond, categories=ORDER).codes
        self.r = df['round'].values.astype(float)


def spec_names(sp):
    nm = ['b00', 'b01', 'log_sigma', 'b_s', 'b_n', 'b_r']
    nm += [f'b_int_{c}' for c in ORDER] if sp.get('cint') else ['b_int']
    nm += [f'kappa_{c}' for c in ORDER] if sp.get('ckap') else ['kappa']
    if sp.get('delta'): nm.append('delta')
    if sp.get('gamma'): nm.append('gamma')
    if sp.get('w') == 'free': nm.append('w')
    return nm


def unpack(x, sp):
    nm = spec_names(sp)
    return {k: x[i] for i, k in enumerate(nm)}, nm


def nll_grad(x, D, sp, GH):
    gx, gw = GH
    pr, nm = unpack(x, sp)
    nc = 7
    sig = np.exp(pr['log_sigma'])
    th = MU0 + np.sqrt(2) * sig * gx
    ci = D.c
    bint = np.array([pr[f'b_int_{c}'] for c in ORDER])[ci] if sp.get('cint') else pr['b_int']
    kapv = np.array([pr[f'kappa_{c}'] for c in ORDER])[ci] if sp.get('ckap') else pr['kappa']
    k = np.where(D.prevD, kapv, 1.0)
    resp = pr['b_n'] * D.xn + pr['b_r'] * D.xr
    eta = bint + pr['b_s'] * D.deg + k * resp
    if sp.get('delta'): eta = eta + pr['delta'] * D.prevD
    if sp.get('gamma'): eta = eta + pr['gamma'] * D.r
    base = np.where(D.r0, pr['b00'], eta)
    slope = np.where(D.r0, pr['b01'], 1.0)
    lin = base[:, None] + slope[:, None] * th[None, :]
    if sp.get('w') is None:
        w = 0.0
    elif sp['w'] == 'free':
        w = pr['w']
    else:
        w = float(sp['w'])
    sgn = 2 * D.y - 1
    if w == 0.0 and sp.get('w') != 'free':
        ll = -np.logaddexp(0.0, -sgn[:, None] * lin)
        p = expit(lin)
        G = D.y[:, None] - p
    else:
        p = expit(lin)
        we = w * D.hasm
        pm = (1 - we)[:, None] * p + (we * D.m)[:, None]
        pm = np.clip(pm, 1e-300, 1 - 1e-16)
        ll = np.where(D.y[:, None] == 1, np.log(pm), np.log1p(-pm))
        dldp = np.where(D.y[:, None] == 1, 1 / pm, -1 / (1 - pm))
        G = dldp * (1 - we)[:, None] * p * (1 - p)
    lp = np.asarray(D.P @ ll) + np.log(gw)[None, :]
    mx = lp.max(1, keepdims=True)
    Li = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
    nll = -Li.sum()
    pi = np.exp(lp - Li[:, None])                       # posterior node weights per player
    Wn = pi[D.pidx]                                     # (N, Q)
    g = (Wn * G).sum(1)
    hq = (Wn * G * (np.sqrt(2) * sig * gx)[None, :]).sum(1)
    hth = (Wn * G * th[None, :]).sum(1)
    grad = {}
    r0 = D.r0; rl = ~r0
    grad['b00'] = g[r0].sum()
    grad['b01'] = hth[r0].sum()
    grad['log_sigma'] = (slope * hq).sum()
    grad['b_s'] = (g * D.deg)[rl].sum()
    grad['b_n'] = (g * k * D.xn)[rl].sum()
    grad['b_r'] = (g * k * D.xr)[rl].sum()
    if sp.get('cint'):
        for q, c in enumerate(ORDER):
            grad[f'b_int_{c}'] = g[rl & (ci == q)].sum()
    else:
        grad['b_int'] = g[rl].sum()
    gk = g * D.prevD * resp
    if sp.get('ckap'):
        for q, c in enumerate(ORDER):
            grad[f'kappa_{c}'] = gk[ci == q].sum()
    else:
        grad['kappa'] = gk.sum()
    if sp.get('delta'): grad['delta'] = (g * D.prevD)[rl].sum()
    if sp.get('gamma'): grad['gamma'] = (g * D.r)[rl].sum()
    if sp.get('w') == 'free':
        dldw = np.where(D.y[:, None] == 1, 1 / pm, -1 / (1 - pm)) * (D.m[:, None] - p) * D.hasm[:, None]
        grad['w'] = (Wn * dldw).sum()
    return nll, -np.array([grad[k_] for k_ in nm])


START = dict(b00=1.7, b01=0.75, log_sigma=np.log(3.4), b_s=-0.27, b_n=0.23, b_r=3.4, b_int=0.5, kappa=0.87,
             delta=0.0, gamma=0.0, w=0.01)


def x0_for(sp, base=None, jitter=None):
    nm = spec_names(sp)
    out = []
    for k in nm:
        if base is not None and k in base and np.isfinite(base[k]):
            out.append(base[k])
        elif k.startswith('b_int_'):
            out.append((base or START).get('b_int', START['b_int']))
        elif k.startswith('kappa_'):
            out.append((base or START).get('kappa', START['kappa']))
        else:
            out.append(START[k])
    x = np.array(out, float)
    if jitter is not None:
        x = x + jitter.normal(0, 0.15, len(x)) * np.maximum(np.abs(x), 0.2)
        if 'w' in nm: x[nm.index('w')] = abs(x[nm.index('w')]) * 0.5 + 0.005
    return x


def bounds_for(sp):
    b = []
    for k in spec_names(sp):
        if k == 'w': b.append((0.0, 0.5))
        elif k == 'log_sigma': b.append((-2.0, 3.0))
        else: b.append((None, None))
    return b


def fit(df, sp, Q=QDEF, starts=3, base=None, mcol='mimit', coding='V1', seed=0):
    D = Data(df, mcol=mcol, coding=coding)
    gx, gw = np.polynomial.hermite.hermgauss(Q); gw = gw / np.sqrt(np.pi)
    GH = (gx, gw)
    rng = np.random.default_rng(seed)
    best = None
    for st in range(starts):
        x0 = x0_for(sp, base, None if st == 0 else rng)
        r = minimize(nll_grad, x0, args=(D, sp, GH), jac=True, method='L-BFGS-B', bounds=bounds_for(sp),
                     options=dict(maxiter=3000, ftol=1e-12, gtol=1e-6))
        if best is None or r.fun < best.fun - 1e-9:
            best = r
    pr, nm = unpack(best.x, sp)
    out = dict(pr); out['sigma'] = float(np.exp(pr['log_sigma']))
    if sp.get('w') not in (None, 'free'): out['w'] = float(sp['w'])
    out.update(nll=float(best.fun), n_obs=D.n, n_players=D.np_, converged=bool(best.success), Q=Q,
               gradmax=float(np.abs(best.jac).max()), npar=len(nm))
    return out


def nll_at(df, sp, pars, Q, mcol='mimit', coding='V1'):
    D = Data(df, mcol=mcol, coding=coding)
    gx, gw = np.polynomial.hermite.hermgauss(Q); gw = gw / np.sqrt(np.pi)
    x = np.array([pars[k] for k in spec_names(sp)], float)
    return float(nll_grad(x, D, sp, (gx, gw))[0])


SPECS = {'M1': {}, 'M1w': {'w': 'free'}, 'M2': {'delta': True}, 'M3': {'gamma': True},
         'M1i': {'cint': True}, 'M1c': {'cint': True, 'ckap': True}}


# ============================================================ simulation parameter points
def pt_from(f, ph, pid, label, model='M1w', cond=None):
    phi = S.PHI0.copy()
    if ph is not None:
        for q, key in enumerate(E.PHI_KEYS):
            v = ph.get(key)
            if v is not None and np.isfinite(v):
                phi[q] = v
    bint = f[f'b_int_{cond}'] if cond is not None and f'b_int_{cond}' in f else f['b_int']
    kap = f[f'kappa_{cond}'] if cond is not None and f'kappa_{cond}' in f else f['kappa']
    betas = [f['b00'], f['b01'], bint, f['b_s'], f['b_n'], f['b_r']]
    if 'gamma' in f and np.isfinite(f.get('gamma', np.nan)):
        betas.append(f['gamma'])
    return dict(grid='empirical', pid=pid, label=label, kappa=float(kap), w=float(f.get('w', 0.0) or 0.0),
                sd_th=float(f['sigma']), mu_th=MU0, phi=[float(v) for v in phi], betas=[float(b) for b in betas])


SI_F = dict(b00=1.807, b01=0.818, b_int=-0.010, b_s=-0.193, b_n=0.370, b_r=1.521, sigma=2.41, kappa=1.0, w=0.0)


# ============================================================ worker tasks (fork; globals PANEL, REC)
def t_fit(args):
    name, sub, model, kw = args
    df = PANEL if sub is None else PANEL[PANEL.cond.isin(sub)] if isinstance(sub, (list, tuple)) else sub
    f = fit(df, SPECS[model] if isinstance(model, str) else model, **kw)
    f['subset'] = name; f['model'] = model if isinstance(model, str) else 'custom'
    return f


def resample(df, rec, rng):
    parts, rparts = [], []
    for cond, dc in df.groupby('cond'):
        gs = dc.group.unique()
        for q, gsel in enumerate(rng.choice(gs, len(gs), replace=True)):
            d = dc[dc.group == gsel].copy(); d['pid'] = d.pid.astype(str) + f'_b{q}'; parts.append(d)
            rparts.append(rec[rec.Group == gsel])
    return pd.concat(parts), pd.concat(rparts)


def t_boot(b):
    rng = np.random.default_rng(7000 + b)
    dfb, rcb = resample(PANEL, REC, rng)
    res = dict(b=b)
    for m in ('M1', 'M1w', 'M3', 'M1i', 'M1c'):
        f = fit(dfb, SPECS[m], starts=1, base=POOLED.get(m))
        for k, v in f.items():
            if isinstance(v, (int, float, np.floating, bool, np.bool_)):
                res[f'{m}_{k}'] = v
    res.update(E.phi_est(rcb))
    return res


def t_profile(args):
    tau_col, wv = args
    sp = {'w': wv} if wv > 0 else {}
    f = fit(PANEL, sp, starts=2, base=POOLED.get('M1'), mcol=tau_col)
    return dict(tau_col=tau_col, w=wv, nll=f['nll'], kappa=f['kappa'], converged=f['converged'])


def t_place(pt):
    rows = []
    for smp in pt.get('samples', ('A', 'B')):
        block = pt['block'] + (0 if smp == 'A' else 1)
        rr = C.evaluate_full(pt, pt.get('N', 10000), block)
        for r in rr:
            r['sample'] = smp; r['label'] = pt['label']
        rows += rr
    return rows


def t_traj(args):
    q, pt, s = args
    kw = W.simkw(pt)
    seeds = np.arange(20000, dtype=np.int64) + 60000 * (9500 + q)
    res = []
    for ss in (s, s - 1):
        res.append(S.simulate_traj(seeds, 0, ss, S.BIG, branch=s, **kw))
    (o1, t1), (o0, t0) = res
    rows = []
    for t in range(15):
        rows.append(dict(label=pt['label'], branch=s, round=t + 1, coop_s=t1[t, 0], coop_sm1=t0[t, 0],
                         coopD_s=t1[t, 1], coopD_sm1=t0[t, 1], coopC_s=t1[t, 2], coopC_sm1=t0[t, 2],
                         deg_s=t1[t, 3], deg_sm1=t0[t, 3], ncd_s=t1[t, 4], ncd_sm1=t0[t, 4],
                         nbind_s=t1[t, 5], nbind_sm1=t0[t, 5], nD=t1[t, 6]))
    d = o1[:, 0] - o0[:, 0]
    rows[-1]['dfin'] = float(d.mean()); rows[-1]['dfin_se'] = float(d.std(ddof=1) / np.sqrt(len(d)))
    return rows


def t_oos(args):
    label, cond, pt, ngames, block = args
    kw = W.simkw(pt)
    seeds = np.arange(ngames, dtype=np.int64) + 60000 * block
    r = S.simulate(seeds, E.SIMCODE[cond], 0, S.BIG, **kw)
    return dict(paramset=label, cond=cond, sim_final=float(r[:, 0].mean()), sim_final_se=float(r[:, 0].std(ddof=1) / np.sqrt(ngames)),
                sim_mean=float(r[:, 1].mean()))


def t_loco_boot(args):
    cond, b = args
    rng = np.random.default_rng(90000 + 100 * ORDER.index(cond) + b)
    dtr = PANEL[PANEL.cond != cond]; rtr = REC[REC.cond != cond]
    dfb, rcb = resample(dtr, rtr, rng)
    f = fit(dfb, SPECS['M3'], starts=1, base=POOLED.get('M3'))
    ph = E.phi_est(rcb)
    pt = pt_from(f, ph, 0, 'x')
    o = t_oos(('loco_M3_boot', cond, pt, 2000, 61000 + 100 * ORDER.index(cond) + b))
    o['b'] = b; o['kappa'] = f['kappa']
    return o


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)


if __name__ == '__main__':
    zpath, OUT, nw, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    quick = len(sys.argv) > 5
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    PANEL, REC, OUTC, FLOW = build_panel2(zpath)
    save(PANEL, 'panel_v2.csv.gz'); save(OUTC, 'observed_rounds_v2.csv'); save(FLOW, 'sample_flow.csv')
    print('panel', PANEL.shape, 'bind', int(PANEL.bind.sum()), 'nb_bot>0 rows', int((PANEL.nb_bot > 0).sum()),
          'prev_bot', int(PANEL.prev_bot.sum()), round(time.time() - t0), flush=True)
    S.simulate(np.arange(4)); S.simulate_traj(np.arange(4), 0, 3, S.BIG, branch=3)
    ctx = mp.get_context('fork')
    POOLED = {}
    # ---------------- 1. quadrature convergence (fixed parameters, then refits)
    # Earlier-panel inputs are absent from the cloud package. Preserve its
    # saved historical comparison instead of pretending to refit that panel.
    historical_nll, historical_fits = load_historical_comparison()
    qrows = historical_nll.to_dict('records')
    print('historical quadrature: using archived cloud comparison', flush=True)
    jobs = [(f'pooled_Q{Q}', None, 'M1', dict(Q=Q)) for Q in (24, 96, 160, 320)]
    jobs += [(f'pooled_Q{Q}', None, 'M3', dict(Q=Q)) for Q in (24, 160, 320)]
    jobs += [(f'pooled_Q{Q}', None, 'M1w', dict(Q=Q)) for Q in (24, 160, 320)]
    with ctx.Pool(min(nw, len(jobs))) as pool:
        qf = pool.map(t_fit, jobs)
    qf = pd.DataFrame(qf)
    for _, r in qf.iterrows():
        if r.subset.startswith('pooled_Q'):
            pars = r.to_dict()
            for Q in (24, 96, 160, 240, 320, 480):
                qrows.append(dict(what=f'{r.model}_fitted_at_Q{r.Q}', Q=Q, nll=nll_at(PANEL, SPECS[r.model], pars, Q)))
    qf = pd.concat([qf, historical_fits], ignore_index=True)
    save(pd.DataFrame(qrows), 'quadrature_nll.csv'); save(qf, 'quadrature_fits.csv')
    print(qf[['subset', 'model', 'Q', 'nll', 'kappa', 'sigma', 'b_r', 'converged', 'gradmax']].to_string(), flush=True)
    print('quadrature', round(time.time() - t0), flush=True)
    # ---------------- 2. point fits at Q = QDEF
    jobs = [('pooled', None, m, {}) for m in ('M1', 'M1w', 'M2', 'M3', 'M1i', 'M1c')]
    jobs += [('pooled_tau0.03', None, 'M1w', dict(mcol='mimit_t003')), ('pooled_tau0.3', None, 'M1w', dict(mcol='mimit_t03'))]
    jobs += [('pooled_V0coding', None, 'M1', dict(coding='V0')), ('pooled_V0coding', None, 'M3', dict(coding='V0'))]
    jobs += [('complete_groups', PANEL[PANEL.dropgrp == 0], 'M1', {}), ('complete_groups', PANEL[PANEL.dropgrp == 0], 'M3', {})]
    jobs += [('baseline_only', list(BASELINE), m, {}) for m in ('M1', 'M1w', 'M3')]
    jobs += [('planner_only', [c for c in ORDER if c not in BASELINE], m, {}) for m in ('M1', 'M1w', 'M3')]
    for c in ORDER:
        jobs += [(c, [c], 'M1', {})]
        jobs += [('loco_' + c, [x for x in ORDER if x != c], m, {}) for m in ('M1', 'M1w', 'M3')]
    with ctx.Pool(min(nw, len(jobs))) as pool:
        fits = pd.DataFrame(pool.map(t_fit, jobs, chunksize=1))
    save(fits, 'fits_v2.csv')
    phis = [dict(E.phi_est(REC), subset='pooled')]
    for c in ORDER:
        phis.append(dict(E.phi_est(REC[REC.cond == c]), subset=c))
        phis.append(dict(E.phi_est(REC[REC.cond != c]), subset='loco_' + c))
    phis.append(dict(E.phi_est(REC[REC.cond.isin(BASELINE)]), subset='baseline_only'))
    phis.append(dict(E.phi_est(REC[~REC.cond.isin(BASELINE)]), subset='planner_only'))
    phis = pd.DataFrame(phis); save(phis, 'phi_v2.csv')
    fx = {(r.subset, r.model): r.to_dict() for _, r in fits.iterrows()}
    for m in SPECS:
        POOLED[m] = fx[('pooled', m)]
    print(fits[fits.subset == 'pooled'][['model', 'nll', 'npar', 'kappa', 'sigma', 'b_s', 'b_n', 'b_r', 'converged', 'gradmax']].to_string(), flush=True)
    print('point fits', round(time.time() - t0), flush=True)
    # ---------------- 3. profile likelihood for w
    wgrid = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 0.01, 0.02, 0.05, 0.1]
    jobs = [(col, wv) for col in ('mimit_t003', 'mimit', 'mimit_t03') for wv in wgrid]
    with ctx.Pool(min(nw, len(jobs))) as pool:
        prof = pd.DataFrame(pool.map(t_profile, jobs, chunksize=1))
    save(prof, 'w_profile.csv')
    print('profile', round(time.time() - t0), flush=True)
    # ---------------- 4. group bootstrap
    Bq = 4 if quick else B
    with ctx.Pool(nw) as pool:
        boots = pd.DataFrame(pool.map(t_boot, range(Bq), chunksize=1))
    save(boots, 'bootstrap_v2.csv')
    print('bootstrap', round(time.time() - t0), flush=True)
    # ---------------- 5. placement
    php = phis.set_index('subset').loc['pooled'].to_dict()
    PH = phis.set_index('subset')
    pts = []
    blk = 12000
    def add(pt, **kw):
        global blk
        pt.update(kw); pt['block'] = blk; blk += 2; pts.append(pt)
    add(pt_from(POOLED['M1w'], php, 0, 'pooled_M1w'))
    add(pt_from(POOLED['M3'], php, 0, 'pooled_M3'))
    add(pt_from(POOLED['M1'], php, 0, 'pooled_M1'))
    for c in ORDER:
        ph_c = PH.loc[c].to_dict() if c != 'static' else php
        add(pt_from(POOLED['M1c'], ph_c, 0, f'M1c_{c}', cond=c))
    for sub in ('baseline_only', 'planner_only'):
        add(pt_from(fx[(sub, 'M1w')], PH.loc[sub].to_dict(), 0, f'{sub}_M1w'))
        add(pt_from(fx[(sub, 'M3')], PH.loc[sub].to_dict(), 0, f'{sub}_M3'))
    add(pt_from(fx[('pooled_V0coding', 'M3')], php, 0, 'V0coding_M3'))
    add(pt_from(fx[('complete_groups', 'M3')], php, 0, 'complete_groups_M3'))
    for kap in np.linspace(0, 1.5, 16):
        for m in ('M1w', 'M3'):
            f = dict(POOLED[m]); f['kappa'] = float(kap)
            add(pt_from(f, php, 0, f'kslice_{m}_{kap:.2f}'))
    # parameter-replacement decomposition between the published simulator (SI) and pooled M1w
    EMP = dict(POOLED['M1w'])
    groups = {'kappa': ['kappa'], 'slopes': ['b_int', 'b_s', 'b_n', 'b_r'], 'sigma': ['sigma'], 'round1': ['b00', 'b01'],
              'phi': ['phi'], 'w': ['w']}
    for g_, keys in groups.items():
        # SI_plus: published simulator with group g_ replaced by the empirical values
        f = dict(SI_F); ph_use = None
        for k_ in keys:
            if k_ == 'phi': ph_use = php
            else: f[k_] = EMP[k_]
        add(pt_from(f, ph_use, 0, f'decomp_SI_plus_{g_}'))
        # EMP_minus: empirical population with group g_ reset to the published values
        f = dict(EMP); ph_use = php
        for k_ in keys:
            if k_ == 'phi': ph_use = None
            else: f[k_] = SI_F[k_]
        add(pt_from(f, ph_use, 0, f'decomp_EMP_minus_{g_}'))
    add(pt_from(dict(SI_F), None, 0, 'decomp_SI'))
    # bootstrap draws: single sample, N = 5000, full family
    for _, r in boots.iterrows():
        ph_b = {k: r[k] for k in E.PHI_KEYS}
        for m in ('M1w', 'M3'):
            f = {k[len(m) + 1:]: r[k] for k in boots.columns if k.startswith(m + '_')}
            pt = pt_from(f, ph_b, 0, f'boot_{m}_{int(r.b)}')
            add(pt, samples=('A',), N=5000)
    for q, p in enumerate(pts):
        p['pid'] = 40000 + q
    json.dump(pts, open(os.path.join(OUT, 'placement_points.json'), 'w'))
    assert max(p['block'] for p in pts) + 1 < 2 ** 32 // 60000 - 1
    with ctx.Pool(nw) as pool:
        pl = pool.map(t_place, sorted(pts, key=lambda p: -len(p.get('samples', 'AB')) * p.get('N', 10000)), chunksize=1)
    pl = pd.DataFrame([r for rr in pl for r in rr]); pl.to_csv(os.path.join(OUT, 'placement.csv.gz'), index=False)
    print('placement', round(time.time() - t0), flush=True)
    # ---------------- 6. matched clock trajectories at empirical parameters
    tp = [pt_from(POOLED['M1w'], php, 0, 'emp_M1w')]
    f = dict(POOLED['M1w']); f['kappa'] = 1.0; tp.append(pt_from(f, php, 0, 'emp_M1w_k1.0'))
    pnc = pt_from(POOLED['M1w'], php, 0, 'emp_M1w_nocap'); pnc['D0'] = 1e6; tp.append(pnc)
    tp.append(pt_from(POOLED['M3'], php, 0, 'emp_M3'))
    tt = [(q, p, s) for q, p in enumerate(tp) for s in range(1, 15)]
    with ctx.Pool(min(nw, len(tt))) as pool:
        tr = pool.map(t_traj, tt, chunksize=1)
    save(pd.DataFrame([r for rr in tr for r in rr]), 'traj_emp.csv')
    print('trajectories', round(time.time() - t0), flush=True)
    # ---------------- 7. out-of-sample prediction
    jobs = []
    si = dict(grid='x', pid=0, kappa=1.0, w=0.0)
    for q, c in enumerate(ORDER):
        blk0 = 62000 + 20 * q
        jobs.append(('SI', c, si, 10000, blk0))
        for m in ('M1', 'M1w', 'M3'):
            jobs.append((f'pooled_{m}', c, pt_from(POOLED[m], php, 0, 'x'), 10000, blk0 + 1))
            jobs.append((f'loco_{m}', c, pt_from(fx[('loco_' + c, m)], PH.loc['loco_' + c].to_dict(), 0, 'x'), 10000, blk0 + 2))
        jobs.append(('M1c_own', c, pt_from(POOLED['M1c'], PH.loc[c].to_dict() if c != 'static' else php, 0, 'x', cond=c), 10000, blk0 + 3))
    with ctx.Pool(min(nw, len(jobs))) as pool:
        oo = pd.DataFrame(pool.map(t_oos, jobs))
    save(oo, 'oos_v2.csv')
    Bl = 2 if quick else 40
    with ctx.Pool(nw) as pool:
        lb = pd.DataFrame(pool.map(t_loco_boot, [(c, b) for c in ORDER for b in range(Bl)], chunksize=1))
    save(lb, 'oos_loco_boot.csv')
    print('done', round(time.time() - t0), flush=True)
