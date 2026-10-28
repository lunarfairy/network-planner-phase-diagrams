"""Build every processed table and the manuscript number file from the raw simulation and estimation outputs.

Usage: python analysis/build_tables.py <raw_dir> <processed_dir>
raw_dir must contain cert/ (cert.csv.gz, cert_points.json, traj.csv), emp/ (outputs of ph_emp3.py),
reward/ (reward.csv.gz), theory/ (theory_first_order.csv) and dense/ (dense.csv.gz, outputs of ph_dense.py).  All numbers quoted in the manuscript are
written to <processed_dir>/numbers.json; verify_numbers.py checks the LaTeX sources against that file.
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import rev_common as RC

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDER = ['static', 'random', 'clustering', 'graphnet', 'encouragement', 'neutral', 'maxconn']
NUM = {}


def num(key, val, nd=4):
    if isinstance(val, (bool, np.bool_)):
        NUM[key] = bool(val)
    elif isinstance(val, (float, np.floating, int, np.integer)):
        NUM[key] = round(float(val), nd) if nd > 0 else int(round(float(val)))
    else:
        NUM[key] = val


def combine(d):
    """Pool samples A and B (equal N) of one point: means, paired differences to E and C, and to the pooled best."""
    a = d[d['sample'] == 'A'].set_index(['planner', 's', 'k'])
    b = d[d['sample'] == 'B'].set_index(['planner', 's', 'k'])
    c = a.copy()
    fam = c.index.get_level_values(0) == 'family'
    for o in ('fin', 'meanc', 'cap'):
        c[o] = (a[o] + b[o]) / 2
        c[o + '_se'] = np.sqrt(a[o + '_se'] ** 2 + b[o + '_se'] ** 2) / 2
        for r in ('E', 'C'):
            c[f'd{r}_{o}'] = (a[f'd{r}_{o}'] + b[f'd{r}_{o}']) / 2
            c[f'd{r}_{o}_se'] = np.sqrt(a[f'd{r}_{o}_se'] ** 2 + b[f'd{r}_{o}_se'] ** 2) / 2
        ib = c.loc[fam, f'dE_{o}'].idxmax()
        c[f'dbest_{o}'] = c[f'dE_{o}'] - c.loc[ib, f'dE_{o}']
        c[f'dbest_{o}_se'] = np.sqrt(c[f'dE_{o}_se'] ** 2 + c.loc[ib, f'dE_{o}_se'] ** 2)
    c = c.reset_index()
    c['sample'] = 'AB'
    return c


def cross(x, y, lev):
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        if (y[i] - lev) * (y[i + 1] - lev) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (lev - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return np.nan


# ====================================================================== certified sweep
cert = pd.read_csv(os.path.join(RAW, 'cert', 'cert.csv.gz'))
cpts = pd.DataFrame(json.load(open(os.path.join(RAW, 'cert', 'cert_points.json'))))
cpts['T'] = cpts['T'].fillna(15).astype(int)
Tmap = dict(zip(cpts.pid, cpts['T']))
rows = []
for pid, d in cert.groupby('pid'):
    T = Tmap[pid]
    out = dict(pid=pid)
    dA, dB, dAB = d[d['sample'] == 'A'], d[d['sample'] == 'B'], combine(d)
    for nm, dd in (('A', dA), ('B', dB), ('AB', dAB)):
        for k, v in RC.classify(dd, T=T).items():
            out[f'{nm}_{k}'] = v
    for eps in (0.0025, 0.01):
        cl = RC.classify(dAB, eps=eps, T=T)
        out[f'AB_phase_eps{eps}'] = cl['phase']; out[f'AB_s_eps{eps}'] = cl['s_star']
    for o in ('meanc', 'cap'):
        cl = RC.classify(dAB, obj=o, T=T)
        out[f'AB_phase_{o}'] = cl['phase']; out[f'AB_s_{o}'] = cl['s_star']; out[f'AB_k_{o}'] = cl['k_star']
    lk = RC.locked(dB, out['A_s_star'], out['A_k_star'], T=T)
    for k, v in lk.items():
        out['lockB_' + k] = v
    fA = dA[dA.planner == 'family']
    fB = dB[dB.planner == 'family'].set_index(['s', 'k'])
    ic = fA.cap.idxmax(); sc, kc = int(fA.loc[ic, 's']), int(fA.loc[ic, 'k'])
    out.update(cap_s=sc, cap_k=kc, lockB_fin_capwin=float(fB.loc[(sc, kc), 'fin']),
               lockB_fin_coopwin=float(fB.loc[(out['A_s_star'], out['A_k_star']), 'fin']))
    enc = dAB[dAB.planner == 'encouragement'].iloc[0]
    out['AB_gap_enc'] = float(-enc['dbest_fin'])
    out['crn_ratio'] = float(dA.crn_var_ratio_EC.iloc[0])
    rows.append(out)
PH = pd.DataFrame(rows).merge(cpts, on='pid')
PH['caploss'] = PH.lockB_fin_coopwin - PH.lockB_fin_capwin
PH['uncertain'] = ((PH.A_phase != PH.B_phase)
                   | (((PH.AB_dE - 0.005).abs() < 2 * PH.AB_dE_se) & PH.AB_phase.isin(['E', 'S']))
                   | (((PH.AB_dC - 0.005).abs() < 2 * PH.AB_dC_se) & PH.AB_phase.isin(['C', 'S'])))
PH['game_T'] = PH.game.apply(lambda v: v[2] if isinstance(v, list) else np.nan)
PH['game_S'] = PH.game.apply(lambda v: v[1] if isinstance(v, list) else np.nan)
PH.drop(columns=['phi', 'game']).to_csv(os.path.join(OUT, 'phase_table_certified.csv'), index=False)

kw = PH[PH.grid == 'kw']
vc = kw.AB_phase.value_counts()
num('n_points', len(PH), 0); num('n_points_kw', len(kw), 0)
NUM['games_cert'] = int(len(cert) * 10000)
for p in ('E', 'S', 'C'):
    num(f'kw_frac_{p}', vc.get(p, 0) / len(kw), 3); num(f'kw_n_{p}', vc.get(p, 0), 0)
num('agree_AB_kw', (kw.A_phase == kw.B_phase).mean(), 3); num('agree_AB_all', (PH.A_phase == PH.B_phase).mean(), 3)
num('n_AB_disagree_kw', (kw.A_phase != kw.B_phase).sum(), 0)
num('uncertain_kw', kw.uncertain.sum(), 0); num('uncertain_all', PH.uncertain.sum(), 0)
sb = PH.A_dE - PH.lockB_dE_fin
num('selbias_dE_mean', sb.mean(), 5); num('selbias_dE_sd', sb.std(), 5)
num('selbias_dC_mean', (PH.A_dC - PH.lockB_dC_fin).mean(), 5)
rc = kw.AB_raw_cls.value_counts()
for p in ('E', 'S', 'C', 'SP', 'P'):
    num(f'kw_raw_{p}', rc.get(p, 0), 0)
num('raw_probation_any_all', PH.AB_raw_cls.isin(['P', 'SP']).sum(), 0)
num('final_probation_all', PH.AB_phase.isin(['P', 'SP']).sum(), 0)
num('crn_ratio_median', PH.crn_ratio.median(), 3)
tol = []
for g, dd in (('kw', kw), ('all', PH)):
    for e in (0.0025, 0.01):
        ct = pd.crosstab(dd.AB_phase, dd[f'AB_phase_eps{e}'])
        for a in ct.index:
            for b in ct.columns:
                tol.append(dict(grid=g, eps=e, from_phase=a, to_phase=b, n=int(ct.loc[a, b])))
        num(f'tol_agree_{g}_{e}', (dd.AB_phase == dd[f'AB_phase_eps{e}']).mean(), 3)
TOL = pd.DataFrame(tol); TOL.to_csv(os.path.join(OUT, 'tolerance_transitions.csv'), index=False)


def tcount(g, e, a, b):
    return int(TOL[(TOL.grid == g) & (TOL.eps == e) & (TOL.from_phase == a) & (TOL.to_phase == b)].n.sum())


for g in ('kw', 'all'):
    num(f'tol_{g}_half_EtoS', tcount(g, 0.0025, 'E', 'S'), 0); num(f'tol_{g}_half_CtoS', tcount(g, 0.0025, 'C', 'S'), 0)
    num(f'tol_{g}_dbl_StoE', tcount(g, 0.01, 'S', 'E'), 0); num(f'tol_{g}_dbl_StoC', tcount(g, 0.01, 'S', 'C'), 0)
for e in (0.0025, 0.01):
    v = kw[f'AB_phase_eps{e}'].value_counts()
    for p in ('E', 'S', 'C'):
        num(f'kw_n_{p}_eps{e}', v.get(p, 0), 0)
w0 = kw[kw.w == 0].sort_values('kappa')
w0[['kappa', 'AB_phase', 'AB_s_star', 'AB_dE', 'AB_dE_se', 'AB_dC', 'AB_dC_se', 'A_phase', 'B_phase', 'AB_raw_cls', 'AB_raw_s',
    'AB_raw_k', 'AB_gap_enc']].to_csv(os.path.join(OUT, 'slice_w0.csv'), index=False)
num('kappa_ES_w0', cross(w0.kappa, w0.AB_dE, 0.005), 2)
w0s = w0[w0.kappa >= 0.7]
num('kappa_SC_w0', cross(w0s.kappa, w0s.AB_dC, 0.005), 2)
num('dE_max_w0', w0.AB_dE.max(), 3); num('dE_max_w0_kappa', w0.loc[w0.AB_dE.idxmax(), 'kappa'], 2)
num('dC_max_kw', kw.AB_dC.max(), 2)
si = kw[(kw.kappa == 1.0) & (kw.w == 0)].iloc[0]
num('SI_phase', si.AB_phase); num('SI_dE', si.AB_dE, 3); num('SI_dE_se', si.AB_dE_se, 4)
for k in (0.6, 0.7, 0.8, 0.9):
    r = w0[np.isclose(w0.kappa, k)].iloc[0]
    num(f'w0_dE_k{k}', r.AB_dE, 4); num(f'w0_dC_k{k}', r.AB_dC, 4); num(f'w0_s_k{k}', r.AB_s_star, 0)
num('encgap_w0_k0', w0.iloc[0].AB_gap_enc, 3)
num('encgap_w0_max_07_11', w0[(w0.kappa >= 0.7) & (w0.kappa <= 1.1)].AB_gap_enc.max(), 3)
num('encgap_k0_w06', kw[(kw.w == 0.6) & (kw.kappa == 0)].iloc[0].AB_gap_enc, 3)
reps = {'kw': kw[kw.w == 0], 'horizon': PH[(PH.grid == 'horizon') & (PH['T'] == 15)], 'sigma': PH[(PH.grid == 'sigma') & (PH.sd_scale == 1)],
        'mu': PH[(PH.grid == 'mu') & (PH.mu_shift == 0)], 'size': PH[(PH.grid == 'size') & (PH.n == 16)]}
RP = pd.concat([d.assign(rep=k)[['rep', 'kappa', 'AB_phase', 'AB_s_star', 'AB_dE', 'AB_dE_se', 'AB_dC']] for k, d in reps.items()])
RP.to_csv(os.path.join(OUT, 'replicates_default.csv'), index=False)
num('rep_phase_agree', float(RP.groupby('kappa').AB_phase.nunique().eq(1).mean()), 3)
num('rep_dE_sd_k1', RP[RP.kappa == 1.0].AB_dE.std(), 4)
num('rep_dE_se_k1', RP[RP.kappa == 1.0].AB_dE_se.mean(), 4)
for k in (0.7, 0.8, 0.9):
    x = RP[np.isclose(RP.kappa, k)].AB_s_star
    num(f'rep_s_range_k{k}', f"{int(x.min())}-{int(x.max())}" if x.min() != x.max() else str(int(x.min())))
num('cap_conc_end_kw', int(((kw.cap_s == 14) & (kw.cap_k == 1000)).sum()), 0)
num('caploss_max_kw', kw.caploss.max(), 3)
r = kw.loc[kw.caploss.idxmax()]; num('caploss_max_kappa', r.kappa, 2); num('caploss_max_w', r.w, 2); num('caploss_max_phase', r.AB_phase)
num('caploss_max_S', kw[kw.AB_phase == 'S'].caploss.max(), 3); num('caploss_mean_S', kw[kw.AB_phase == 'S'].caploss.mean(), 3)
for k in (0.5, 0.7):
    rr = w0[np.isclose(w0.kappa, k)].iloc[0]; num(f'caploss_w0_k{k}', rr.lockB_fin_coopwin - rr.lockB_fin_capwin, 3); num(f'phase_w0_k{k}', rr.AB_phase)
mc = kw.AB_phase_meanc.value_counts()
for p in ('E', 'S', 'C'):
    num(f'kw_meanc_n_{p}', mc.get(p, 0), 0)
num('kw_S_to_E_under_meanc', int(((kw.AB_phase == 'S') & (kw.AB_phase_meanc == 'E')).sum()), 0)
num('kw_S_to_C_under_meanc', int(((kw.AB_phase == 'S') & (kw.AB_phase_meanc == 'C')).sum()), 0)
cp = kw.AB_phase_cap.value_counts()
for p in ('E', 'S', 'C', 'P'):
    num(f'kw_cap_n_{p}', cp.get(p, 0), 0)
hz = PH[PH.grid == 'horizon']
H = hz[['kappa', 'T', 'AB_phase', 'AB_s_star']].copy(); H['L'] = H['T'] - 1 - H.AB_s_star
H.to_csv(os.path.join(OUT, 'horizon_window.csv'), index=False)
h8 = H[np.isclose(H.kappa, 0.8)]
num('L_k08_T8_30', f"{int(h8[h8['T'] >= 8].L.min())}-{int(h8[h8['T'] >= 8].L.max())}")
num('L_k08_T15', int(h8[h8['T'] == 15].L.iloc[0]), 0); num('L_k08_T5', int(h8[h8['T'] == 5].L.iloc[0]), 0)
STR = PH[PH.grid.isin(['horizon', 'sigma', 'mu', 'size', 'phi', 'comp_all', 'comp_add', 'comp_del', 'unanimity'])][
    ['pid', 'grid', 'kappa', 'w', 'T', 'sd_scale', 'mu_shift', 'n', 'phi_AD', 'lam', 'AB_phase', 'AB_s_star', 'AB_dE', 'AB_dC', 'uncertain']]
STR.to_csv(os.path.join(OUT, 'robustness_strips.csv'), index=False)
ph_ = PH[PH.grid == 'phi']; num('phi_AD_phase_constant', bool(ph_.groupby('kappa').AB_phase.nunique().eq(1).all()))
cd = PH[(PH.grid == 'comp_del') & (PH.lam == 0.25)]; num('compdel025_E_upto', float(cd[cd.AB_phase == 'E'].kappa.max()), 2)
ca = PH[(PH.grid == 'comp_all')]; num('compall_P_points', int((ca.AB_phase == 'P').sum()), 0)
cad = PH[(PH.grid == 'comp_add') & (PH.lam == 0.25)]; num('compadd025_S_from', float(cad[cad.AB_phase == 'S'].kappa.min()), 2)
s16 = PH[(PH.grid == 'size') & (PH.n == 64)]; num('size64_S_from', float(s16[s16.AB_phase == 'S'].kappa.min()), 2); num('size64_C_from', float(s16[s16.AB_phase == 'C'].kappa.min()), 2)
sg = PH[(PH.grid == 'sigma') & (PH.sd_scale == 0.25)]; num('sigma025_nE', int((sg.AB_phase == 'E').sum()), 0)
un = PH[PH.grid == 'unanimity']
num('unan_s', f"{int(un.AB_s_star.min())}" if un.AB_s_star.nunique() == 1 else f"{int(un.AB_s_star.min())}-{int(un.AB_s_star.max())}")
num('unan_dE_min', un.AB_dE.min(), 2); num('unan_dE_max', un.AB_dE.max(), 2); num('unan_dC_min', un.AB_dC.min(), 2); num('unan_dC_max', un.AB_dC.max(), 2)
cs = cert[(cert.planner == 'family') & (cert['sample'] == 'A')].merge(cpts[cpts.grid == 'size'][['pid', 'kappa', 'n']], on='pid')
cs = cs[(cs.k == 1000) & cs.s.isin([0, 14])].copy()
cs['final_deg'] = cs.dens * (cs.n - 1); cs['rec_per_node'] = cs.nrec / cs.n
cs[['kappa', 'n', 's', 'fin', 'final_deg', 'rec_per_node']].to_csv(os.path.join(OUT, 'size_network_stats.csv'), index=False)
c8 = cs[np.isclose(cs.kappa, 0.8)]
for n in (16, 64):
    for s_, lab in ((0, 'E'), (14, 'C')):
        r = c8[(c8.n == n) & (c8.s == s_)].iloc[0]
        num(f'size_deg_{lab}_n{n}', r.final_deg, 1); num(f'size_rec_{lab}_n{n}', r.rec_per_node, 1)
gp = PH[PH.grid == 'game'].copy()
gp['Tg'] = gp.game_T; gp['Sg'] = gp.game_S; gp['TS'] = gp.Tg - gp.Sg
grows = []
for w, g in gp.groupby('w'):
    if w == 0:
        num('game_w0_C', int((g.AB_phase == 'C').sum()), 0); num('game_w0_n', len(g), 0)
        num('game_w0_nonC_Smax', float(g[g.AB_phase != 'C'].Sg.max()), 2)
        continue
    for col, lab, ge in (('TS', 'E', True), ('Tg', 'E', True), ('Sg', 'E', False), ('TS', 'C', False), ('Tg', 'C', False)):
        ths = np.arange(-3, 5.01, 0.25)
        accs = [(((g[col] >= t) if ge else (g[col] <= t)) == (g.A_phase == lab)).mean() for t in ths]
        t_ = ths[int(np.argmax(accs))]
        pred = (g[col] >= t_) if ge else (g[col] <= t_)
        pdq = g[(g.Tg > 1) & (g.Sg < 0)]
        predq = (pdq[col] >= t_) if ge else (pdq[col] <= t_)
        grows.append(dict(w=w, target=lab, variable=col, rule=('>=' if ge else '<='), threshold=t_, acc_A=max(accs),
                          acc_B=(pred == (g.B_phase == lab)).mean(), acc_B_PDquadrant=(predq == (pdq.B_phase == lab)).mean(),
                          majority_B=max((g.B_phase == lab).mean(), 1 - (g.B_phase == lab).mean())))
    num(f'game_TSmixed_w{w}', int(sum(gg.AB_phase.nunique() > 1 for _, gg in g.groupby('TS'))), 0)
    num(f'game_TSlevels_w{w}', int(g.TS.nunique()), 0)
GP = pd.DataFrame(grows); GP.to_csv(os.path.join(OUT, 'game_plane_rules.csv'), index=False)
for _, r in GP.iterrows():
    key = f"game_{r.target}_{r.variable}_w{r.w}"
    num(key + '_thr', r.threshold, 2); num(key + '_accB', r.acc_B, 3); num(key + '_base', r.majority_B, 3); num(key + '_accPD', r.acc_B_PDquadrant, 3)
gp[['w', 'Tg', 'Sg', 'AB_phase', 'A_phase', 'B_phase', 'AB_s_star', 'AB_dE', 'AB_dC']].to_csv(os.path.join(OUT, 'game_plane.csv'), index=False)

# ====================================================================== theory (end game)
th = pd.read_csv(os.path.join(RAW, 'theory', 'theory_first_order.csv'))
e = th[th.t == 14].dropna(subset=['sim_end_diff']).copy()
e['err'] = e.M_total - e.sim_end_diff
num('th_n', len(e), 0); num('th_sign_agree', int((np.sign(e.M_total) == np.sign(e.sim_end_diff)).sum()), 0)
num('th_mae_all', e.err.abs().mean(), 4); num('th_max_err', e.err.abs().max(), 3)
nb = e[e.sim_end_diff.abs() < 0.02]; num('th_n_near', len(nb), 0); num('th_mae_near', nb.err.abs().mean(), 4)
num('th_corr', np.corrcoef(e.M_total, e.sim_end_diff)[0, 1], 3)
num('th_slope_sim_on_theory', np.polyfit(e.M_total, e.sim_end_diff, 1)[0], 2)
mm = e[np.sign(e.M_total) != np.sign(e.sim_end_diff)].iloc[0]
num('th_mismatch_kappa', mm.kappa, 2); num('th_mismatch_w', mm.w, 2); num('th_mismatch_M', mm.M_total, 4)
num('th_mismatch_sim', mm.sim_end_diff, 4); num('th_mismatch_sim_se', mm.sim_end_se, 4)
r = e.loc[e.err.abs().idxmax()]; num('th_maxerr_kappa', r.kappa, 2); num('th_maxerr_w', r.w, 2)
e.to_csv(os.path.join(OUT, 'theory_endgame.csv'), index=False)
th.to_csv(os.path.join(OUT, 'theory_all_rounds.csv'), index=False)
neg = th[(th.kappa >= 0.7) & (th.kappa <= 0.9) & (th.w == 0)]
num('th_Mt_neg_all_rounds_07_09', bool((neg.M_total < 0).all()))

# ====================================================================== matched-branch mechanism
TR = pd.concat([pd.read_csv(os.path.join(RAW, 'cert', 'traj.csv')), pd.read_csv(os.path.join(RAW, 'emp', 'traj_emp.csv'))])
# additive group contributions (cD/n, cC/n per game) from the re-run of the same branches (ph_branches.py);
# overall cooperation in that file reproduces traj.csv / traj_emp.csv exactly
_BR = pd.read_csv(os.path.join(RAW, 'branches', 'branches.csv'))[['label', 'branch', 'round', 'popD_s', 'popD_sm1', 'popC_s', 'popC_sm1']]
TR = TR.merge(_BR, on=['label', 'branch', 'round'], how='left')
assert TR.popD_s.notna().all()
TR.to_csv(os.path.join(OUT, 'branch_trajectories.csv'), index=False)
mech = []
for lab, d in TR.groupby('label'):
    for s_, g in d.groupby('branch'):
        g = g.set_index('round'); r1 = s_ + 1; nD = g.loc[r1, 'nD']; n = 16
        mech.append(dict(label=lab, s=s_,
                         imm=g.loc[r1, 'coop_s'] - g.loc[r1, 'coop_sm1'],
                         immD=g.loc[r1, 'popD_s'] - g.loc[r1, 'popD_sm1'],
                         immC=g.loc[r1, 'popC_s'] - g.loc[r1, 'popC_sm1'],
                         fin=g.loc[15, 'dfin'], fin_se=g.loc[15, 'dfin_se'],
                         finD=g.loc[15, 'popD_s'] - g.loc[15, 'popD_sm1'],
                         finC=g.loc[15, 'popC_s'] - g.loc[15, 'popC_sm1'],
                         cum=(g.loc[r1:, 'coop_s'] - g.loc[r1:, 'coop_sm1']).sum(),
                         ddeg_fin=g.loc[15, 'deg_s'] - g.loc[15, 'deg_sm1'],
                         dbind=(g.loc[r1:, 'nbind_s'] - g.loc[r1:, 'nbind_sm1']).sum()))
MECH = pd.DataFrame(mech)
assert (MECH.imm - MECH.immD - MECH.immC).abs().max() < 1e-9
MECH.to_csv(os.path.join(OUT, 'mechanism_branches.csv'), index=False)
for lab, key in (('orig_k0.8', 'o08'), ('orig_k1.0', 'o10'), ('orig_k0.6', 'o06'), ('emp_M1w', 'emp'), ('emp_M1w_k1.0', 'empk1'),
                 ('emp_M1w_nocap', 'empnc'), ('orig_k0.8_nocap', 'o08nc'), ('emp_M3', 'empM3')):
    g = MECH[(MECH.label == lab) & (MECH.s <= 10)]
    for col in ('imm', 'immD', 'immC', 'fin', 'cum', 'finD', 'finC', 'ddeg_fin'):
        num(f'mech_{key}_{col}_min', g[col].min(), 4); num(f'mech_{key}_{col}_max', g[col].max(), 4); num(f'mech_{key}_{col}_mean', g[col].mean(), 4)
    num(f'mech_{key}_sumfin_1_10', g.fin.sum(), 4)
    num(f'mech_{key}_fin_se', g.fin_se.mean(), 4)

# ====================================================================== empirical
E = os.path.join(RAW, 'emp')
fits = pd.read_csv(os.path.join(E, 'fits_v2.csv')); fits.to_csv(os.path.join(OUT, 'fits.csv'), index=False)
boots = pd.read_csv(os.path.join(E, 'bootstrap_v2.csv'))
F = {(r.subset, r.model): r for _, r in fits.iterrows()}


def pct(x):
    return np.percentile(np.asarray(x, float), [2.5, 97.5])


est = []
for m in ('M1', 'M3', 'M2', 'M1i'):
    r = F[('pooled', m)]
    for p in ('kappa', 'b_int', 'b_s', 'b_n', 'b_r', 'sigma', 'b00', 'b01', 'gamma', 'delta'):
        if p in r and np.isfinite(r[p]) and f'{m}_{p}' in boots:
            lo, hi = pct(boots[f'{m}_{p}'])
            est.append(dict(model=m, param=p, est=r[p], lo=lo, hi=hi))
r = F[('pooled', 'M1c')]
for c in ORDER:
    for p in (f'kappa_{c}', f'b_int_{c}'):
        lo, hi = pct(boots[f'M1c_{p}']); est.append(dict(model='M1c', param=p, est=r[p], lo=lo, hi=hi))
EST = pd.DataFrame(est); EST.to_csv(os.path.join(OUT, 'estimates.csv'), index=False)
for _, r in EST.iterrows():
    k = f"est_{r.model}_{r.param}"; num(k, r.est, 3); num(k + '_lo', r.lo, 3); num(k + '_hi', r.hi, 3)
num('n_obs', int(F[('pooled', 'M1')].n_obs), 0); num('n_players', int(F[('pooled', 'M1')].n_players), 0)
num('B_boot', len(boots), 0)
num('boot_converged_all', bool(all(boots[f'{m}_converged'].all() for m in ('M1', 'M1w', 'M3', 'M1i', 'M1c'))))
num('n_boot_fits', 5 * len(boots), 0)
num('w_hat', F[('pooled', 'M1w')].w, 6); num('w_boot_zero_frac', (boots.M1w_w == 0).mean(), 3)
for sub, m, key in (('pooled_V0coding', 'M1', 'V0'), ('complete_groups', 'M1', 'complete'), ('baseline_only', 'M1', 'baseline'), ('planner_only', 'M1', 'planner')):
    r = F[(sub, m)]
    for p in ('kappa', 'b_r', 'sigma', 'b_n'):
        num(f'fit_{key}_{p}', r[p], 3)
num('nll_M1', F[('pooled', 'M1')].nll, 2)
tests = {'intercepts': (('pooled', 'M1'), ('pooled', 'M1i'), 6), 'kappa_c': (('pooled', 'M1i'), ('pooled', 'M1c'), 6),
         'joint': (('pooled', 'M1'), ('pooled', 'M1c'), 12), 'delta': (('pooled', 'M1'), ('pooled', 'M2'), 1), 'trend': (('pooled', 'M1'), ('pooled', 'M3'), 1)}
TT = []
for k, (a, b, df) in tests.items():
    lr = 2 * (F[a].nll - F[b].nll); TT.append(dict(test=k, stat=lr, df=df, p=chi2.sf(lr, df)))
    num(f'LR_{k}', lr, 1); NUM[f'LRp_{k}'] = float(f'{chi2.sf(lr, df):.2g}')


def wald(cols, model):
    X = boots[cols].values
    est_ = np.array([F[('pooled', model)][c.split('_', 1)[1]] for c in cols])
    Cm = np.eye(len(cols))[1:] - np.eye(len(cols))[0]
    dd = Cm @ est_; V = Cm @ np.cov(X.T) @ Cm.T
    W = float(dd @ np.linalg.solve(V, dd)); return W, chi2.sf(W, len(cols) - 1)


Wk = wald([f'M1c_kappa_{c}' for c in ORDER], 'M1c'); Wi = wald([f'M1i_b_int_{c}' for c in ORDER], 'M1i')
TT += [dict(test='wald_kappa_c_cluster', stat=Wk[0], df=6, p=Wk[1]), dict(test='wald_intercepts_cluster', stat=Wi[0], df=6, p=Wi[1])]
num('Wald_kappa_c', Wk[0], 1); NUM['Waldp_kappa_c'] = float(f'{Wk[1]:.2g}'); num('Wald_int', Wi[0], 1); NUM['Waldp_int'] = float(f'{Wi[1]:.2g}')
pd.DataFrame(TT).to_csv(os.path.join(OUT, 'tests.csv'), index=False)
prof = pd.read_csv(os.path.join(E, 'w_profile.csv'))
PR = []
for col, g in prof.groupby('tau_col'):
    g = g.sort_values('w'); d = 2 * (g.nll - g.nll.min()).values; w = g.w.values
    i = int(np.argmax(d > 2.706)); ub = float(np.interp(2.706, [d[i - 1], d[i]], [w[i - 1], w[i]]))
    tau = {'mimit_t003': 0.03, 'mimit': 0.1, 'mimit_t03': 0.3}[col]
    PR.append(dict(tau=tau, w_hat=float(w[np.argmin(d)]), w_upper95=ub, LR_at_w001=float(d[np.isclose(w, 0.01)][0])))
    num(f'w_upper_tau{tau}', ub, 4); num(f'w_LR001_tau{tau}', float(d[np.isclose(w, 0.01)][0]), 1)
pd.DataFrame(PR).to_csv(os.path.join(OUT, 'w_profile_summary.csv'), index=False)
prof.to_csv(os.path.join(OUT, 'w_profile.csv'), index=False)
q = pd.read_csv(os.path.join(E, 'quadrature_nll.csv')); q.to_csv(os.path.join(OUT, 'quadrature_nll.csv'), index=False)
qf = pd.read_csv(os.path.join(E, 'quadrature_fits.csv')); qf.to_csv(os.path.join(OUT, 'quadrature_fits.csv'), index=False)
qo = q[q.what == 'old_panel_old_M1_params'].set_index('Q').nll
num('quad_old_Q24', qo.loc[24], 3); num('quad_old_Q160', qo.loc[160], 3); num('quad_old_Q320', qo.loc[320], 3)
qn = q[q.what == 'M1_fitted_at_Q160'].set_index('Q').nll
num('quad_new_Q160', qn.loc[160], 3); num('quad_new_Q320', qn.loc[320], 3); num('quad_new_Q24', qn.loc[24], 3)
for sub in ('oldpanel_Q160', 'oldpanel_Q24'):
    r = qf[(qf.subset == sub)].iloc[0]
    for p in ('kappa', 'sigma', 'b_r', 'nll'):
        num(f'{sub}_{p}', r[p], 4 if p == 'kappa' else 3)
r1, r2 = qf[(qf.subset == 'pooled_Q160') & (qf.model == 'M1')].iloc[0], qf[(qf.subset == 'pooled_Q320') & (qf.model == 'M1')].iloc[0]
num('quad_refit_dkappa', abs(r1.kappa - r2.kappa), 5); num('quad_refit_dsigma', abs(r1.sigma - r2.sigma), 4)
r3a, r3b = qf[(qf.subset == 'pooled_Q160') & (qf.model == 'M3')].iloc[0], qf[(qf.subset == 'pooled_Q320') & (qf.model == 'M3')].iloc[0]
num('quad_refit_M3_dkappa', abs(r3a.kappa - r3b.kappa), 5)
phi = pd.read_csv(os.path.join(E, 'phi_v2.csv')); phi.to_csv(os.path.join(OUT, 'phi.csv'), index=False)
pp = phi[phi.subset == 'pooled'].iloc[0]
for k in ('phi_del_refD', 'phi_del_refC', 'phi_add_refD', 'phi_add_refC'):
    num(f'{k}', pp[k], 3); lo, hi = pct(boots[k]); num(f'{k}_lo', lo, 3); num(f'{k}_hi', hi, 3)
fl = pd.read_csv(os.path.join(E, 'sample_flow.csv'))
P2 = pd.read_csv(os.path.join(E, 'panel_v2.csv.gz'))
sf = fl.groupby('cond').agg(node_rounds=('human', 'size'), human=('human', 'sum'), bind=('bind', 'sum'))
sf['bot_controlled'] = sf.node_rounds - sf.human; sf['analysed'] = sf.human - sf.bind
sf['rows_with_bot_neighbour'] = P2[P2.bind == 0].groupby('cond').nb_bot.apply(lambda x: int((x > 0).sum()))
sf['groups'] = P2.groupby('cond').group.nunique(); sf['participants'] = P2.groupby('cond').pid.nunique()
sf = sf.loc[ORDER]; sf.to_csv(os.path.join(OUT, 'sample_flow.csv'))
num('flow_bot', int(sf.bot_controlled.sum()), 0); num('flow_human', int(sf.human.sum()), 0); num('flow_bind', int(sf.bind.sum()), 0)
num('flow_botnb_rows', int(sf.rows_with_bot_neighbour.sum()), 0); num('flow_node_rounds', int(sf.node_rounds.sum()), 0)
num('flow_groups', int(sf.groups.sum()), 0)
pl = pd.read_csv(os.path.join(E, 'placement.csv.gz'))
_d2 = os.path.join(RAW, 'decomp2', 'decomp2.csv')
if os.path.exists(_d2):   # finer replacement of individual response coefficients (ph_decomp2.py)
    pl = pd.concat([pl, pd.read_csv(_d2)], ignore_index=True)
prow = []
for lab, d in pl.groupby('label'):
    for smp in sorted(d['sample'].unique()):
        c_ = RC.classify(d[d['sample'] == smp]); c_.update(label=lab, sample=smp); prow.append(c_)
    if set(d['sample']) == {'A', 'B'}:
        dd = combine(d.drop(columns=['label']))
        for o in ('fin', 'meanc', 'cap'):
            c_ = RC.classify(dd, obj=o); c_.update(label=lab, sample='AB_' + o); prow.append(c_)
PLc = pd.DataFrame(prow); PLc.to_csv(os.path.join(OUT, 'placement_classes.csv'), index=False)
bt = PLc[PLc.label.str.startswith('boot_')].copy(); bt['model'] = bt.label.str.split('_').str[1]
bt.to_csv(os.path.join(OUT, 'placement_bootstrap.csv'), index=False)
for m, g in bt.groupby('model'):
    num(f'boot_{m}_nS', int((g.phase == 'S').sum()), 0); num(f'boot_{m}_n', len(g), 0)
    num(f'boot_{m}_s11', int((g.s_star == 11).sum()), 0)
    lo, hi = np.percentile(g.dE, [2.5, 97.5]); num(f'boot_{m}_dE_lo', lo, 3); num(f'boot_{m}_dE_hi', hi, 3)
    lo, hi = np.percentile(g.dC, [2.5, 97.5]); num(f'boot_{m}_dC_lo', lo, 3); num(f'boot_{m}_dC_hi', hi, 3)
nonboot = PLc[~PLc.label.str.startswith('boot_') & PLc['sample'].isin(['A', 'B'])]
nonboot.to_csv(os.path.join(OUT, 'placement_points_classes.csv'), index=False)
cond_ok = nonboot[nonboot.label.str.startswith('M1c_')]
num('cond_all_S', bool((cond_ok.phase == 'S').all())); num('cond_s_values', ','.join(sorted(set(cond_ok.s_star.astype(int).astype(str)))))
num('cond_dE_min', cond_ok.dE.min(), 3); num('cond_dE_max', cond_ok.dE.max(), 3)
for lab in ('pooled_M1w', 'pooled_M3', 'pooled_M1', 'baseline_only_M1w', 'baseline_only_M3', 'planner_only_M1w', 'planner_only_M3', 'V0coding_M3', 'complete_groups_M3'):
    a = nonboot[(nonboot.label == lab) & (nonboot['sample'] == 'A')].iloc[0]
    b = nonboot[(nonboot.label == lab) & (nonboot['sample'] == 'B')].iloc[0]
    num(f'place_{lab}_phase', a.phase if a.phase == b.phase else f'{a.phase}/{b.phase}'); num(f'place_{lab}_s', int(a.s_star), 0)
MO = []
for lab in ('pooled_M1w', 'pooled_M3'):
    d = pl[(pl.label == lab) & (pl['sample'] == 'B')]
    for name, sel in (('switch_s11', (d.planner == 'family') & (d.s == 11) & (d.k == 1000)), ('exclusion', (d.planner == 'family') & (d.s == 0)),
                      ('conciliation', (d.planner == 'family') & (d.s == 14) & (d.k == 1000)), ('encouragement', d.planner == 'encouragement'),
                      ('static', d.planner == 'static')):
        r = d[sel].iloc[0]
        MO.append(dict(model=lab, rule=name, **{c: r[c] for c in ('fin', 'fin_se', 'meanc', 'meanc_se', 'cap', 'cap_se', 'dE_fin', 'dE_fin_se',
                                                                   'dE_meanc', 'dE_meanc_se', 'dE_cap', 'dE_cap_se', 'dC_fin', 'dC_fin_se')}))
MO = pd.DataFrame(MO); MO.to_csv(os.path.join(OUT, 'empirical_multiobjective.csv'), index=False)
for _, r in MO.iterrows():
    k = f"mo_{r.model}_{r.rule}"
    for c in ('fin', 'meanc', 'cap', 'dE_fin', 'dE_fin_se', 'dE_meanc', 'dE_meanc_se', 'dE_cap', 'dE_cap_se', 'dC_fin', 'dC_fin_se'):
        num(f'{k}_{c}', r[c], 3)
for lab in ('pooled_M1w', 'pooled_M3'):
    for o in ('meanc', 'cap'):
        r = PLc[(PLc.label == lab) & (PLc['sample'] == 'AB_' + o)].iloc[0]
        num(f'place_{lab}_{o}_phase', r.phase); num(f'place_{lab}_{o}_s', int(r.s_star), 0); num(f'place_{lab}_{o}_k', int(r.k_star), 0)
ks = nonboot[nonboot.label.str.startswith('kslice_')].copy()
ks['model'] = ks.label.str.split('_').str[1]; ks['kappa'] = ks.label.str.split('_').str[2].astype(float)
ks.to_csv(os.path.join(OUT, 'empirical_kslice.csv'), index=False)
for m in ('M1w', 'M3'):
    g = ks[ks.model == m]
    both = g.groupby('kappa').phase.agg(lambda x: ''.join(sorted(set(x))))
    num(f'kslice_{m}_S_from', float(both[both == 'S'].index.min()), 2)
    num(f'kslice_{m}_C_any', bool((g.phase == 'C').any()))
    num(f'kslice_{m}_E_upto', float(both[both == 'E'].index.max()), 2)
dc = nonboot[nonboot.label.str.startswith('decomp')].copy(); dc.to_csv(os.path.join(OUT, 'decomposition.csv'), index=False)
for lab, g in dc.groupby('label'):
    a = g[g['sample'] == 'A'].iloc[0]; b = g[g['sample'] == 'B'].iloc[0]
    key = lab.replace('decomp2_', 'dec2_').replace('decomp_', 'dec_')
    num(key + '_phase', a.phase if a.phase == b.phase else a.phase + '/' + b.phase)
    num(key + '_dC', (a.dC + b.dC) / 2, 3); num(key + '_dE', (a.dE + b.dE) / 2, 3)
oo = pd.read_csv(os.path.join(E, 'oos_v2.csv')); lb = pd.read_csv(os.path.join(E, 'oos_loco_boot.csv'))
obs = pd.read_csv(os.path.join(E, 'observed_rounds_v2.csv'))
fin = obs[obs['round'] == 14].groupby(['cond', 'group']).agg(coop=('coop', 'mean'), c16=('coop_all16', 'mean')).reset_index()
ob = fin.groupby('cond').agg(obs=('coop', 'mean'), obs_se=('coop', 'sem'), obs16=('c16', 'mean'), n_groups=('coop', 'size'))
tab = oo.pivot(index='cond', columns='paramset', values='sim_final').loc[ORDER].join(ob)
pi = lb.groupby('cond').sim_final.quantile([0.025, 0.975]).unstack().loc[ORDER]
tab['loco_M3_lo'] = pi[0.025]; tab['loco_M3_hi'] = pi[0.975]
tab.to_csv(os.path.join(OUT, 'oos_table.csv'))
osum = []
for p in ('SI', 'pooled_M1', 'pooled_M3', 'loco_M1', 'loco_M3', 'M1c_own'):
    err = (tab[p] - tab.obs).abs()
    osum.append(dict(paramset=p, MAE=err.mean(), max_err=err.max(), max_cond=err.idxmax(), rho=spearmanr(tab[p], tab.obs).correlation,
                     MAE_ex_maxconn=err.drop('maxconn').mean(), MAE_ex_graphnet=err.drop('graphnet').mean(), MAE_all16=(tab[p] - tab.obs16).abs().mean()))
    for c in ORDER:
        num(f'oos_err_{p}_{c}', err[c], 3)
OS = pd.DataFrame(osum); OS.to_csv(os.path.join(OUT, 'oos_summary.csv'), index=False)
for _, r in OS.iterrows():
    for c in ('MAE', 'max_err', 'rho', 'MAE_ex_maxconn', 'MAE_ex_graphnet', 'MAE_all16'):
        num(f'oos_{r.paramset}_{c}', r[c], 3)
    num(f'oos_{r.paramset}_max_cond', r.max_cond)
for c in ORDER:
    num(f'obs_{c}', tab.loc[c, 'obs'], 3); num(f'pred_loco_M3_{c}', tab.loc[c, 'loco_M3'], 3); num(f'pred_SI_{c}', tab.loc[c, 'SI'], 3)
    num(f'pred_loco_M3_{c}_lo', tab.loc[c, 'loco_M3_lo'], 3); num(f'pred_loco_M3_{c}_hi', tab.loc[c, 'loco_M3_hi'], 3)
worse = [c for c in ORDER if abs(tab.loc[c, 'loco_M3'] - tab.loc[c, 'obs']) > abs(tab.loc[c, 'SI'] - tab.loc[c, 'obs'])]
num('oos_loco_worse_than_SI', len(worse), 0); num('oos_loco_worse_conds', ','.join(worse))
rng = np.random.default_rng(3)
maes, maesSI = [], []
for b in sorted(lb.b.unique()):
    pr = lb[lb.b == b].set_index('cond').sim_final.loc[ORDER].values
    o = np.array([rng.choice(fin[fin.cond == c].coop.values, (fin.cond == c).sum()).mean() for c in ORDER])
    maes.append(np.abs(pr - o).mean()); maesSI.append(np.abs(tab['SI'].values - o).mean())
maes, maesSI = np.array(maes), np.array(maesSI)
num('oos_boot_B', len(maes), 0)
lo, hi = np.percentile(maesSI - maes, [2.5, 97.5]); num('oos_gain_lo', lo, 3); num('oos_gain_hi', hi, 3); num('oos_gain_pos_frac', (maesSI - maes > 0).mean(), 3)
lo, hi = np.percentile(maes, [2.5, 97.5]); num('oos_locoM3_boot_lo', lo, 3); num('oos_locoM3_boot_hi', hi, 3)
rwf = os.path.join(RAW, 'reward', 'reward.csv.gz')
if os.path.exists(rwf):
    rw = pd.read_csv(rwf)
    rr = []
    for pid, d in rw.groupby('pid'):
        out = dict(pid=pid, label=d.label.iloc[0] if isinstance(d.label.iloc[0], str) else '')
        A_, B_ = d[d['sample'] == 'A'], d[d['sample'] == 'B'].set_index(['s', 'k'])
        for o in ('fin', 'meanc', 'cap', 'dcap', 'dutil'):
            i = A_[o].idxmax(); s_, k_ = int(A_.loc[i, 's']), int(A_.loc[i, 'k'])
            out[f'{o}_s'] = s_; out[f'{o}_k'] = k_; out[f'{o}_finB'] = float(B_.loc[(s_, k_), 'fin'])
        rr.append(out)
    RW = pd.DataFrame(rr).merge(cpts[['pid', 'kappa', 'w']], on='pid', how='left').merge(PH[['pid', 'AB_phase']], on='pid', how='left')
    for o in ('cap', 'dcap', 'dutil', 'meanc'):
        RW[f'loss_{o}'] = RW.fin_finB - RW[f'{o}_finB']
    RW.to_csv(os.path.join(OUT, 'reward_objectives.csv'), index=False)
    kwr = RW[RW.pid < 50000]
    for o in ('cap', 'dcap', 'dutil'):
        num(f'rw_{o}_concend', int(((kwr[f'{o}_s'] == 14) & (kwr[f'{o}_k'] == 1000)).sum()), 0)
        num(f'rw_{o}_lossmax', kwr[f'loss_{o}'].max(), 3)
        num(f'rw_{o}_lossmax_S', kwr[kwr.AB_phase == 'S'][f'loss_{o}'].max(), 3)
    for _, r in RW[RW.pid >= 50000].iterrows():
        for o in ('dcap', 'dutil', 'cap'):
            num(f'rw_{r.label}_{o}_rule', f"{int(r[o + '_s'])},{int(r[o + '_k'])}"); num(f'rw_{r.label}_{o}_loss', r.fin_finB - r[o + '_finB'], 3)


# ---------------------------------------------------------------- dense kappa grid near the boundaries (ph_dense.py)
_df = os.path.join(RAW, 'dense', 'dense.csv.gz')
if os.path.exists(_df):
    _dd = pd.read_csv(_df); _dp = pd.DataFrame(json.load(open(os.path.join(RAW, 'dense', 'dense_points.json'))))
    _res = []
    for _pid, _d in _dd.groupby('pid'):
        _o = dict(pid=_pid)
        for _nm, _x in (('A', _d[_d['sample'] == 'A']), ('B', _d[_d['sample'] == 'B']), ('AB', combine(_d))):
            _cl = RC.classify(_x); _o[_nm + '_phase'] = _cl['phase']; _o[_nm + '_s'] = _cl['s_star']
            if _nm == 'AB':
                _o.update(dE=_cl['dE'], dE_se=_cl['dE_se'], dC=_cl['dC'], dC_se=_cl['dC_se'], raw_cls=_cl['raw_cls'])
        _res.append(_o)
    DN = pd.DataFrame(_res).merge(_dp[['pid', 'kappa', 'w']], on='pid').sort_values(['w', 'kappa'])
    DN['uncertain'] = (DN.A_phase != DN.B_phase) | ((DN.dE - 0.005).abs() < 2 * DN.dE_se) | ((DN.dC - 0.005).abs() < 2 * DN.dC_se)
    DN.to_csv(os.path.join(OUT, 'dense_boundary.csv'), index=False)
    num('dense_n', len(DN), 0); num('dense_AB_agree', int((DN.A_phase == DN.B_phase).sum()), 0); num('dense_uncertain', int(DN.uncertain.sum()), 0)
    for _w, _tag in ((0.0, 'w0'), (0.1, 'w01')):
        _g = DN[DN.w == _w]
        _k = _g.kappa.values; _ph = _g.AB_phase.values
        _iS = [i for i in range(len(_ph)) if _ph[i] != 'E'][0]; _iC = [i for i in range(len(_ph)) if _ph[i] == 'C'][0]
        num(f'dense_{_tag}_lastE', _k[_iS - 1], 2); num(f'dense_{_tag}_firstS', _k[_iS], 2)
        num(f'dense_{_tag}_lastS', _k[_iC - 1], 2); num(f'dense_{_tag}_firstC', _k[_iC], 2)
        num(f'dense_{_tag}_cross_dE', cross(_k, _g.dE.values, 0.005), 3)
        num(f'dense_{_tag}_cross_dC', cross(_k, _g.dC.values, 0.005), 3)
        _sS = _g[_g.AB_phase == 'S'].AB_s
        num(f'dense_{_tag}_smin', _sS.min(), 0); num(f'dense_{_tag}_smax', _sS.max(), 0)

# ---------------------------------------------------------------- theory roots and pair statistics
def _roots(x, y):
    out = []
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] < 0:
            out.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return out


_TE = pd.read_csv(os.path.join(OUT, 'theory_endgame.csv')); _TA = pd.read_csv(os.path.join(OUT, 'theory_all_rounds.csv'))
for _w, _tag in ((0.0, 'w0'), (0.1, 'w01')):
    _g = _TE[_TE.w == _w].sort_values('kappa')
    for _j, _r in enumerate(_roots(_g.kappa.values, _g.M_total.values)):
        num(f'th_root_M_{_tag}_{_j}', _r, 3)
    for _j, _r in enumerate(_roots(_g.kappa.values, _g.sim_end_diff.values)):
        num(f'th_root_sim_{_tag}_{_j}', _r, 3)
_g1 = _TA[(_TA.w == 0) & (_TA.t == 1)].sort_values('kappa')
for _j, _r in enumerate(_roots(_g1.kappa.values, _g1.M_total.values)):
    num(f'th_root_M1_w0_{_j}', _r, 3)
_b = dict(bs=0.193, bn=0.370, br=1.521)
for _t, _tag in ((14, 'dense'), (1, 'sparse')):
    _r = _TA[(_TA.kappa == 1.0) & (_TA.w == 0) & (_TA.t == _t)].iloc[0]
    for _c in ('v_j', 'v_i', 'r_j', 'r_i'):
        num(f'th_pair_{_tag}_{_c}', _r[_c], 3)
    for _c in ('d_j', 'd_i'):
        num(f'th_pair_{_tag}_{_c}', _r[_c], 1)
    _kc = (_r.v_j * _b['bs'] + _r.v_i * (_b['bs'] + _b['br'] * _r.r_i / (_r.d_i + 1))) / (_r.v_j * (_b['bn'] + _b['br'] * (1 - _r.r_j) / (_r.d_j + 1)))
    num(f'th_kcrit_{_tag}', _kc, 2)

# ---------------------------------------------------------------- language-model pilot (Supplementary Note 10)
_pf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llm_pilot', 'llm_elicitation_responses.json')
if os.path.exists(_pf):
    import re as _re
    _L = pd.DataFrame(json.load(open(_pf, encoding='utf-8')))
    _L['ans'] = _L.text.fillna('').str.upper().str.extract(r'(COOPERATE|DEFECT|ACCEPT|REJECT)')[0]
    num('pilot_n', len(_L), 0); num('pilot_parsed', int(_L.ans.notna().sum()), 0); num('pilot_parse_pct', 100 * _L.ans.notna().mean(), 1)
    num('pilot_reject', int((_L.ans == 'REJECT').sum()), 0)

json.dump(NUM, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=str)
print('numbers', len(NUM))
