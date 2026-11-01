import re
"""Write the LaTeX tables (main Table 1 and Supplementary Tables 1-9) from the processed tables.

Usage: python analysis/make_tables.py <processed_dir> <raw_dir> <latex_dir>
"""
import sys, os, json
import numpy as np
import pandas as pd

P, RAW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
N = json.load(open(os.path.join(P, 'numbers.json')))
ORDER = ['static', 'random', 'clustering', 'graphnet', 'encouragement', 'neutral', 'maxconn']
CL = {'static': 'Static network', 'random': 'Random recommendations', 'clustering': 'Cooperative clustering', 'graphnet': 'GraphNet planner$^a$',
      'encouragement': 'Encouragement planner', 'neutral': 'Neutral planner', 'maxconn': 'Maximum connectivity'}


def f(x, d=3):
    if isinstance(x, str):
        return x
    s = f'{x:.{d}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


def ci(key, d=3):
    return f"{f(N[key], d)} [{f(N[key + '_lo'], d)}, {f(N[key + '_hi'], d)}]"


def pval(p):
    if p < 1e-3:
        m, e = f'{p:.1e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'
    return f'{p:.3f}'


# ------------------------------------------------------------------ Table 1
rows = [('$\\kap$ (M1)', '1 (fixed)', ci('est_M1_kappa')),
        ('$\\kap$ (M3, round trend)', '--', ci('est_M3_kappa')),
        ('$\\beta_0$ (later-round intercept)', '$-0.010$', ci('est_M1_b_int')),
        ('$\\beta_s$ (degree)', '$-0.193$', ci('est_M1_b_s')),
        ('$\\beta_n$ (cooperating neighbours)', '0.370', ci('est_M1_b_n')),
        ('$\\beta_r$ (share of cooperating neighbours)', '1.521', ci('est_M1_b_r')),
        ('$\\sigma_\\theta$ (spread of dispositions)', '2.41', ci('est_M1_sigma')),
        ('$\\beta_0\'$ (round-1 intercept)', '1.807', ci('est_M1_b00')),
        ('$\\beta_1\'$ (round-1 disposition loading)', '0.818', ci('est_M1_b01')),
        ('Round trend $\\gamma$ (M3)', '--', ci('est_M3_gamma')),
        ('Imitation share $w$ ($\\tau=0.03$, 0.1, 0.3)', '0', f"0 (profile 95\\% bounds {f(min(N['w_upper_tau0.1'], N['w_upper_tau0.03'], N['w_upper_tau0.3']), 4)}--{f(max(N['w_upper_tau0.1'], N['w_upper_tau0.03'], N['w_upper_tau0.3']), 4)})"),
        ('$\\phi(-|D)$', '0.774', ci('phi_del_refD')), ('$\\phi(-|C)$', '0.085', ci('phi_del_refC')),
        ('$\\phi(+|D)$', '0.287', ci('phi_add_refD')), ('$\\phi(+|C)$', '0.909', ci('phi_add_refC'))]
tests = [('Round trend (M3 vs M1), LR, 1 d.f.', f"{N['LR_trend']:.1f}", pval(N['LRp_trend'])),
         ('Shift after defection (M2 vs M1), LR, 1 d.f.', f"{N['LR_delta']:.1f}", pval(N['LRp_delta'])),
         ('Condition intercepts (M1i vs M1), LR, 6 d.f.', f"{N['LR_intercepts']:.1f}", pval(N['LRp_intercepts'])),
         ('Condition $\\kap^{(c)}$ (M1c vs M1i), LR, 6 d.f.', f"{N['LR_kappa_c']:.1f}", pval(N['LRp_kappa_c'])),
         ('Condition intercepts, group-bootstrap Wald, 6 d.f.', f"{N['Wald_int']:.1f}", pval(N['Waldp_int'])),
         ('Condition $\\kap^{(c)}$, group-bootstrap Wald, 6 d.f.', f"{N['Wald_kappa_c']:.1f}", pval(N['Waldp_kappa_c']))]
t = ['\\begin{tabular}{lll}', '\\toprule', 'Parameter & Published simulator & Re-estimated (pooled)\\\\', '\\midrule']
t += [f'{a} & {b} & {c}\\\\' for a, b, c in rows]
t += ['\\midrule', 'Test & Statistic & $P$\\\\', '\\midrule'] + [f'{a} & {b} & {c}\\\\' for a, b, c in tests]
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(OUT, 'table1.tex'), 'w').write('\n'.join(t) + '\n')

# ------------------------------------------------------------------ Supplementary tables
INF, GE, LE = '$\\infty$', '$\\ge$', '$\\le$'
S = []
TABS = {}


def table(label, caption, header, body, spec):
    global S
    S = []
    TABS[label] = S
    S.append('\\begin{table}[h]\\centering\\small')
    S.append(f'\\caption{{{caption}}}\\label{{{label}}}')
    S.append(f'\\begin{{tabular}}{{{spec}}}\\toprule')
    S.append(header + '\\\\\\midrule')
    S += [r + '\\\\' for r in body]
    S.append('\\bottomrule\\end{tabular}\\end{table}\n')


PH = pd.read_csv(os.path.join(P, 'phase_table_certified.csv'))
grids = [('Main plane', 'kw', '$\\kap\\times w\\in\\{0,0.05,\\dots,0.6\\}$'), ('Game plane', 'game', '$T\\in[0,3]\\times S\\in[-2,1]$ (step 0.25) $\\times w\\in\\{0,0.3,0.6\\}$'),
         ('Acceptance $\\phi(+|D)$', 'phi', '$\\kap\\times\\phi(+|D)\\in\\{0.05,0.15,\\dots,0.95\\}$'), ('Horizon', 'horizon', '$\\kap\\times H\\in\\{5,8,10,15,20,25,30\\}$'),
         ('Spread of dispositions', 'sigma', '$\\kap\\times\\sigma_\\theta/2.41\\in\\{0.25,0.5,1,1.5,2\\}$'), ('Mean disposition', 'mu', '$\\kap\\times\\mu_\\theta+\\{-1,-0.5,0,0.5,1\\}$'),
         ('Group size', 'size', '$\\kap\\times n\\in\\{8,16,32,64\\}$, mean degree 4.5'), ('All acceptance', 'comp_all', '$\\kap\\times\\lambda\\in\\{0.1,0.25,0.5,0.75\\}$'),
         ('Addition acceptance', 'comp_add', '$\\kap\\times\\lambda\\in\\{0.25,0.5\\}$'), ('Deletion acceptance', 'comp_del', '$\\kap\\times\\lambda\\in\\{0.25,0.5\\}$'),
         ('Unanimity rule', 'unanimity', '$w\\in\\{0,0.05,\\dots,0.6\\}$')]
body = []
for name, g, desc in grids:
    d = PH[PH.grid == g]
    vc = d.AB_phase.value_counts()
    body.append(f"{name} & {desc} & {len(d)} & {(d.A_phase == d.B_phase).mean()*100:.1f} & {int(d.uncertain.sum())} & "
                + ' / '.join(str(vc.get(k, 0)) for k in ('E', 'S', 'C', 'P')))
body.append(f"Total & & {len(PH)} & {(PH.A_phase == PH.B_phase).mean()*100:.1f} & {int(PH.uncertain.sum())} & "
            + ' / '.join(str((PH.AB_phase == k).sum()) for k in ('E', 'S', 'C', 'P')))
table('stab:grids', 'Parameter grids of the certified sweep. Unless varied, parameters take the published values ($\\kap=1$ except where varied, $w=0$, $H=15$, $n=16$, donation game). $\\kap$ grids use 16 values in $[0,1.5]$. A=B, percentage of points at which samples A and B give the same class; uncertain, points flagged as in Methods; classes from pooled A+B estimates (E/S/C/P; mixed probation never occurs as simplest class).',
      'Grid & Varied parameters & Points & A=B (\\%) & Uncertain & E / S / C / P', body, 'llrrrl')
TOL = pd.read_csv(os.path.join(P, 'tolerance_transitions.csv'))
body = []
for g, gn in (('kw', 'Main plane'), ('all', 'All grids')):
    for e in (0.0025, 0.01):
        d = TOL[(TOL.grid == g) & (TOL.eps == e) & (TOL.from_phase != TOL.to_phase) & (TOL.n > 0)]
        tr = ', '.join(f'{a}$\\to${b}: {n}' for a, b, n in zip(d.from_phase, d.to_phase, d.n))
        body.append(f"{gn} & {e} & {N[f'tol_agree_{g}_{e}']*100:.1f} & {tr}")
table('stab:tol', 'Class changes when the tolerance is halved or doubled from $\\epsilon=0.005$ (pooled A+B estimates).', 'Grid & $\\epsilon$ & Unchanged (\\%) & Transitions from the default class', body, 'llrp{8.5cm}')
RP = pd.read_csv(os.path.join(P, 'replicates_default.csv'))
body = []
for k in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
    d = RP[np.isclose(RP.kappa, k)].set_index('rep')
    body.append(f'{k:g} & ' + ' & '.join(f"{d.loc[r, 'AB_phase']} ({int(d.loc[r, 'AB_s_star'])}; {d.loc[r, 'AB_dE']:.4f})" for r in ('kw', 'horizon', 'sigma', 'mu', 'size')))
table('stab:rep', 'Independent replicates of the default parameters ($w=0$, $H=15$, $n=16$, published dispositions) contained in five grids with disjoint seeds. Entries: class ($s^\\star$; $\\Delta_E$). The standard error of $\\Delta_E$ is about 0.0008.',
      '$\\kap$ & Main plane & Horizon & Spread & Mean & Size', body, 'llllll')
V = pd.read_csv(os.path.join(RAW, 'legacy', 'simulator_validation.csv'))
body = [f"{r.kappa:g} & {r.w:g} & {r.planner} & {r.ref_final:.3f} ({r.ref_se:.3f}) & {r.numba_final:.3f} ({r.numba_se:.3f}) & {r.z:.2f}" for r in V.itertuples()]
table('stab:val', 'Implementation check: compiled simulator against the NumPy reference implementation (final-round cooperation, s.e. in parentheses; reference 600 games, compiled 8,000 games). Both implement the same published model, so this checks the port, not the model.',
      '$\\kap$ & $w$ & Planner & Reference & Compiled & $z$', body, 'rrlllr')
e = pd.read_csv(os.path.join(P, 'theory_endgame.csv'))
body = []
for w, g in e.groupby('w'):
    err = (g.M_total - g.sim_end_diff).abs()
    body.append(f"{w:g} & {len(g)} & {int((np.sign(g.M_total) == np.sign(g.sim_end_diff)).sum())} & {err.mean():.4f} & {err.max():.4f} & {np.corrcoef(g.M_total, g.sim_end_diff)[0, 1]:.3f}")
body.append(f"All & {N['th_n']} & {N['th_sign_agree']} & {N['th_mae_all']:.4f} & {N['th_max_err']:.4f} & {N['th_corr']:.3f}")
table('stab:theory', 'Accuracy of the summed single-tie value $M_{14}$ against the simulated effect of the last conciliation step (40,000 paired games per point). Near the boundary (29 points with $|$effect$|<0.02$) the mean absolute error is ' + f"{N['th_mae_near']:.4f}.",
      '$w$ & Points & Same sign & Mean abs. error & Max. abs. error & Correlation', body, 'rrrrrr')
M = pd.read_csv(os.path.join(P, 'mechanism_branches.csv'))
LAB = {'orig_k0.6': 'published, $\\kap=0.6$', 'orig_k0.8': 'published, $\\kap=0.8$', 'orig_k0.8_nocap': 'published, $\\kap=0.8$, no capital limit',
       'orig_k1.0': 'published, $\\kap=1.0$', 'emp_M1w': 're-estimated (M1w)', 'emp_M1w_k1.0': 're-estimated, $\\kap=1$',
       'emp_M1w_nocap': 're-estimated, no capital limit', 'emp_M3': 're-estimated (M3)'}
body = []
for lab, name in LAB.items():
    g = M[(M.label == lab) & (M.s <= 10)]
    body.append(f"{name} & {f(g.imm.mean(), 4)} & {f(g.immD.mean(), 4)} & {f(g.immC.mean(), 4)} & {f(g.cum.mean(), 3)} & {f(g.fin.mean(), 4)} & {f(g.fin.sum(), 3)} & {f(g.ddeg_fin.mean(), 3)}")
table('stab:mech', 'Matched branches: effect of one extra conciliation step at planner steps $s\\le10$ (means over $s$; 20,000 paired games per step; s.e. of the final-round effect about 0.0007). Immediate, round $s+1$; split into the contributions of round-$s$ defectors and cooperators; cumulative, summed over rounds $s+1,\\dots,15$; final, round 15; $\\Delta$ degree, final mean degree.',
      'Population & Immediate & Defectors & Cooperators & Cumulative & Final & Final, sum & $\\Delta$ degree', body, 'lrrrrrrr')
SF = pd.read_csv(os.path.join(P, 'sample_flow.csv')).set_index('cond').loc[ORDER]
obs = pd.read_csv(os.path.join(P, 'oos_table.csv')).set_index('cond').loc[ORDER]
body = [f"{CL[c].replace('$^a$', '')} & {int(r.groups)} & {int(r.participants)} & {int(r.node_rounds)} & {int(r.bot_controlled)} & {int(r.human)} & {int(r.bind)} & {int(r.analysed)} & {int(r.rows_with_bot_neighbour)} & {obs.loc[c, 'obs']:.3f} & {obs.loc[c, 'obs16']:.3f}" for c, r in SF.iterrows()]
body.append(f"Total & {int(SF.groups.sum())} & {int(SF.participants.sum())} & {int(SF.node_rounds.sum())} & {int(SF.bot_controlled.sum())} & {int(SF.human.sum())} & {int(SF['bind'].sum())} & {int(SF.analysed.sum())} & {int(SF.rows_with_bot_neighbour.sum())} & & ")
table('stab:flow', 'Sample flow. Node-rounds, 16 nodes $\\times$ 15 rounds $\\times$ groups; bot, node-rounds of dropped-out players controlled by bots (absent from the choice files, actions decoded from the edge-type log); human, recorded decisions; capital, decisions forced by the capital constraint (excluded); bot neighbour, analysed decisions with at least one bot-controlled neighbour (coded as defection in the earlier panel). Final cooperation over human participants (as analysed) and over all 16 nodes.',
      'Condition & Groups & Players & Node-rounds & Bot & Human & Capital & Analysed & Bot neighbour & Final (human) & Final (16)', body, 'lrrrrrrrrrr')
q = pd.read_csv(os.path.join(P, 'quadrature_nll.csv')).pivot_table(index='what', columns='Q', values='nll')
QN = {'old_panel_old_M1_params': 'Earlier panel, earlier 24-node M1 estimates', 'M1_fitted_at_Q24': 'Revised panel, M1 fitted with 24 nodes',
      'M1_fitted_at_Q160': 'Revised panel, M1 fitted with 160 nodes', 'M1_fitted_at_Q320': 'Revised panel, M1 fitted with 320 nodes',
      'M3_fitted_at_Q160': 'Revised panel, M3 fitted with 160 nodes'}
cols = [c for c in (24, 48, 96, 160, 240, 320) if c in q.columns]
body = [f"{QN[w]} & " + ' & '.join(('--' if not np.isfinite(q.loc[w, c]) else f'{q.loc[w, c]:.3f}') for c in cols) for w in QN if w in q.index]
qf = pd.read_csv(os.path.join(P, 'quadrature_fits.csv'))
for sub, name in (('oldpanel_Q24', 'Earlier panel, refit 24 nodes'), ('oldpanel_Q160', 'Earlier panel, refit 160 nodes'), ('pooled_Q24', 'Revised panel, refit 24 nodes'),
                  ('pooled_Q160', 'Revised panel, refit 160 nodes'), ('pooled_Q320', 'Revised panel, refit 320 nodes')):
    r = qf[(qf.subset == sub) & (qf.model == 'M1')].iloc[0]
    body.append(f"{name} (M1): NLL {r.nll:.3f}, $\\hat\\kap$ {r.kappa:.4f}, $\\hat\\sigma_\\theta$ {r.sigma:.3f}, $\\hat\\beta_r$ {r.b_r:.3f} & \\multicolumn{{{len(cols)}}}{{l}}{{}}")
table('stab:quad', 'Quadrature convergence. Upper block, negative log-likelihood at fixed parameters against the number of Gauss--Hermite nodes $Q$ ($Q=48$ evaluated only for the earlier panel). Lower block, refitted M1 estimates. The earlier panel coded bot-controlled neighbours as defectors; its 24-node fit is the estimate of an earlier version of this analysis.',
      'Parameters & ' + ' & '.join(f'$Q$={c}' for c in cols), body, 'p{6.2cm}' + 'r' * len(cols))
fits = pd.read_csv(os.path.join(P, 'fits.csv'))
EST = pd.read_csv(os.path.join(P, 'estimates.csv'))
body = []
for c in ORDER:
    k = EST[(EST.model == 'M1c') & (EST.param == f'kappa_{c}')].iloc[0]; b = EST[(EST.model == 'M1c') & (EST.param == f'b_int_{c}')].iloc[0]
    body.append(f"{CL[c]} & {k.est:.3f} [{k.lo:.3f}, {k.hi:.3f}] & {f(b.est)} [{f(b.lo)}, {f(b.hi)}]")
table('stab:cond', 'Condition-specific responsiveness and intercepts from M1c (shared slopes $\\beta_s=' + f"{fits[(fits.subset=='pooled')&(fits.model=='M1c')].b_s.iloc[0]:.3f}" + '$, $\\beta_n=' + f"{fits[(fits.subset=='pooled')&(fits.model=='M1c')].b_n.iloc[0]:.3f}" + '$, $\\beta_r=' + f"{fits[(fits.subset=='pooled')&(fits.model=='M1c')].b_r.iloc[0]:.3f}" + '$, $\\sigma_\\theta=' + f"{fits[(fits.subset=='pooled')&(fits.model=='M1c')].sigma.iloc[0]:.3f}" + '$); 95\\% group-bootstrap intervals. $^a$Planner condition simulated with the encouragement table in prediction.',
      'Condition & $\\kap^{(c)}$ & Intercept $\\beta_0^{(c)}$', body, 'lll')
WP = pd.read_csv(os.path.join(P, 'w_profile.csv'))
body = []
for col, tau in (('mimit_t003', 0.03), ('mimit', 0.1), ('mimit_t03', 0.3)):
    g = WP[WP.tau_col == col].sort_values('w'); d = 2 * (g.nll - g.nll.min())
    body.append(f'{tau:g} & ' + ' & '.join(f'{v:.2f}' for v in d) + f" & {N[f'w_upper_tau{tau}']:.4f}")
ws = sorted(WP.w.unique())
table('stab:wprof', 'Profile likelihood of the imitation share $w$ (all other M1w parameters refitted at each fixed $w$): $2\\Delta$NLL relative to the maximum, and the upper bound where $2\\Delta$NLL $=2.71$ (95\\% point of the $\\frac12\\chi^2_0+\\frac12\\chi^2_1$ mixture). The bound assumes conditionally independent decisions; all 200 group-bootstrap estimates are $\\hat w=0$.',
      '$\\tau$ & ' + ' & '.join(f'{w:g}' for w in ws) + ' & Bound', body, 'r' + 'r' * len(ws) + 'r')
PC = pd.read_csv(os.path.join(P, 'placement_points_classes.csv'))
NAMES = [('pooled_M1', 'Pooled M1'), ('pooled_M1w', 'Pooled M1w'), ('pooled_M3', 'Pooled M3')] + [(f'M1c_{c}', f'M1c, {CL[c]}') for c in ORDER] + \
        [('baseline_only_M1w', 'Baseline conditions only, M1w'), ('baseline_only_M3', 'Baseline conditions only, M3'), ('planner_only_M1w', 'Planner conditions only, M1w'),
         ('planner_only_M3', 'Planner conditions only, M3'), ('complete_groups_M3', 'Groups without dropouts, M3'), ('V0coding_M3', 'Earlier neighbour coding, M3')]
fsub = {('baseline_only', 'M1'): 'baseline_only', ('planner_only', 'M1'): 'planner_only'}
body = []
for lab, name in NAMES:
    a = PC[(PC.label == lab) & (PC['sample'] == 'A')].iloc[0]; b = PC[(PC.label == lab) & (PC['sample'] == 'B')].iloc[0]
    body.append(f"{name.replace('$^a$', '')} & {a.phase} & {b.phase} & {int(a.s_star)} & {a.dE:.3f} & {b.dE:.3f} & {a.dC:.3f}")
table('stab:place', 'Placement of empirical parameter vectors (final-round objective). Class and $s^\\star$ selected on sample A; gains on A and on the independent sample B (B gains are for the B-selected best rule; the locked A rule gives the same values here because both samples select $s^\\star$). Pooled bootstrap: class S in ' + f"{N['boot_M1w_nS']}/200 (M1w) and {N['boot_M3_nS']}/200 (M3) draws.",
      'Parameter vector & Class A & Class B & $s^\\star$ & $\\Delta_E$ (A) & $\\Delta_E$ (B) & $\\Delta_C$ (A)', body, 'lllrrrr')
DC = pd.read_csv(os.path.join(P, 'decomposition.csv'))
RC_ = '$\\beta_0,\\beta_s,\\beta_n,\\beta_r$'
DN = [('decomp_SI', 'Published simulator')] + [(f'decomp_SI_plus_{g}', f'Published, re-estimated {n}') for g, n in (('kappa', '$\\kap$'), ('slopes', RC_), ('sigma', '$\\sigma_\\theta$'), ('round1', 'round-1 coefficients'), ('phi', 'acceptance'), ('w', '$w$'))] + \
     [(f'decomp2_SI_plus_{g}', f'Published, re-estimated {n}') for g, n in (('b_int', '$\\beta_0$'), ('b_s', '$\\beta_s$'), ('b_n', '$\\beta_n$'), ('b_r', '$\\beta_r$'), ('b_int_s_n', '$\\beta_0,\\beta_s,\\beta_n$'))] + \
     [(f'decomp_EMP_minus_{g}', f'Re-estimated, published {n}') for g, n in (('kappa', '$\\kap$'), ('slopes', RC_), ('sigma', '$\\sigma_\\theta$'), ('round1', 'round-1 coefficients'), ('phi', 'acceptance'), ('w', '$w$'))] + \
     [(f'decomp2_EMP_minus_{g}', f'Re-estimated, published {n}') for g, n in (('b_int', '$\\beta_0$'), ('b_s', '$\\beta_s$'), ('b_n', '$\\beta_n$'), ('b_r', '$\\beta_r$'), ('b_int_s_n', '$\\beta_0,\\beta_s,\\beta_n$'))]
body = []
for lab, name in DN:
    g = DC[DC.label == lab]; a = g[g['sample'] == 'A'].iloc[0]; b = g[g['sample'] == 'B'].iloc[0]
    body.append(f"{name} & {a.phase}/{b.phase} & {int(a.s_star)} & {(a.dE + b.dE) / 2:.3f} & {abs((a.dC + b.dC) / 2) if abs((a.dC + b.dC) / 2) < 5e-4 else (a.dC + b.dC) / 2:.3f}")
table('stab:decomp', 'Parameter-replacement decomposition between the published model and the re-estimated M1w model. One block of parameters, or one later-round response coefficient, is exchanged at a time ($\\beta_0$ intercept, $\\beta_s$ degree, $\\beta_n$ number and $\\beta_r$ share of cooperating neighbours). Class on samples A/B; gains averaged over both samples; 10,000 games per sample.',
      'Model & Class & $s^\\star$ & $\\Delta_E$ & $\\Delta_C$', body, 'llrrr')
RW = pd.read_csv(os.path.join(P, 'reward_objectives.csv'))
kw = RW[RW.pid < 50000]
body = []
for o, name in (('meanc', 'Mean cooperation over rounds'), ('cap', 'Mean final capital'), ('dcap', 'Discounted per-round mean capital'), ('dutil', 'Discounted capital minus rejection penalty')):
    conc = int(((kw[o + '_s'] == 14) & (kw[o + '_k'] == 1000)).sum())
    loss = kw['loss_' + o]
    em = RW[RW.pid >= 50000].set_index('label')
    body.append(f"{name} & {conc} & {loss.max():.3f} & {loss.mean():.3f} & {kw[kw.AB_phase=='S']['loss_'+o].max():.3f} & " +
                ' & '.join(f"({int(em.loc[l, o + '_s'])}, {(INF if em.loc[l, o + '_k'] == 1000 else int(em.loc[l, o + '_k']))}) {em.loc[l, 'fin_finB'] - em.loc[l, o + '_finB']:.3f}" for l in ('pooled_M1w', 'pooled_M3')))
table('stab:reward', 'Alternative objectives on the main plane (208 points) and at the re-estimated population. Winners (sample-A argmax, no tolerance) selected on sample A and evaluated on sample B; winner C, number of points at which conciliation throughout is the winner; loss, final-round cooperation of the final-round winner minus that of the objective\'s winner. Last two columns, winning rule $(s,\\ell)$ and loss at the pooled M1w and M3 estimates.',
      'Objective & Winner C (points) & Max. loss & Mean loss & Max. loss in S & M1w & M3', body, 'lrrrrll')
T3 = pd.read_csv(os.path.join(P, 'oos_table.csv')).set_index('cond').loc[ORDER]
body = [f"{CL[c]} ({int(r.n_groups)}) & {r.obs:.3f} ({r.obs_se:.3f}) & {r.SI:.3f} & {r.pooled_M1:.3f} & {r.pooled_M3:.3f} & {r.loco_M1:.3f} & {r.loco_M3:.3f} [{r.loco_M3_lo:.3f}, {r.loco_M3_hi:.3f}] & {r.M1c_own:.3f}" for c, r in T3.iterrows()]
OS = pd.read_csv(os.path.join(P, 'oos_summary.csv')).set_index('paramset')
body.append('Mean absolute error & & ' + ' & '.join(f"{OS.loc[p, 'MAE']:.3f}" for p in ('SI', 'pooled_M1', 'pooled_M3', 'loco_M1', 'loco_M3', 'M1c_own')))
body.append('MAE without maximum connectivity & & ' + ' & '.join(f"{OS.loc[p, 'MAE_ex_maxconn']:.3f}" for p in ('SI', 'pooled_M1', 'pooled_M3', 'loco_M1', 'loco_M3', 'M1c_own')))
body.append('MAE without GraphNet & & ' + ' & '.join(f"{OS.loc[p, 'MAE_ex_graphnet']:.3f}" for p in ('SI', 'pooled_M1', 'pooled_M3', 'loco_M1', 'loco_M3', 'M1c_own')))
body.append('Spearman $\\rho$ & & ' + ' & '.join(f"{OS.loc[p, 'rho']:.2f}" for p in ('SI', 'pooled_M1', 'pooled_M3', 'loco_M1', 'loco_M3', 'M1c_own')))
table('stab:oos', 'Prediction of final-round cooperation in the seven human conditions (10,000 simulated games per entry; observed group means with s.e.; groups in parentheses). LOCO, parameters estimated without the predicted condition, with 95\\% intervals from 40 group-bootstrap refits; M1c own, condition parameters from M1c (in-sample). The improvement in MAE of LOCO M3 over the published simulator has a 95\\% interval of ' + f"{N['oos_gain_lo']:.3f}--{N['oos_gain_hi']:.3f}. $^a$Simulated with the published encouragement table.",
      'Condition & Observed & Published & Pooled M1 & Pooled M3 & LOCO M1 & LOCO M3 & M1c own', body, 'lllllllll'[:8])
GPR = pd.read_csv(os.path.join(P, 'game_plane_rules.csv'))
VN = {'TS': '$T-S$', 'Tg': '$T$', 'Sg': '$S$'}
body = [f"{r.w:g} & {'exclusion' if r.target == 'E' else 'conciliation throughout'} & {VN[r.variable]} {GE if r.rule == '>=' else LE} {r.threshold:g} & {r.acc_A*100:.1f} & {r.acc_B*100:.1f} & {r.majority_B*100:.1f} & {r.acc_B_PDquadrant*100:.1f}" for r in GPR.itertuples()]
table('stab:game', 'Threshold rules on the game plane. Thresholds selected on sample A (step 0.25); accuracy on A (in-sample) and on the independent sample B, the majority-class baseline on B, and accuracy on B within the prisoner\'s-dilemma quadrant ($T>1$, $S<0$).',
      '$w$ & Target class & Rule & Acc. A & Acc. B & Baseline & Acc. B (PD)', body, 'rllrrrr')
DNS = pd.read_csv(os.path.join(P, 'dense_boundary.csv'))
def _dcell(r):
    cls = r.AB_phase + ('' if r.A_phase == r.B_phase else f' ({r.A_phase}/{r.B_phase})')
    return f"{cls} & {int(r.AB_s)} & {r.dE:.4f} ({r.dE_se:.4f}) & {r.dC:.4f} ({r.dC_se:.4f})" + (r' $\dagger$' if r.uncertain else '')
_g0 = DNS[DNS.w == 0].set_index('kappa'); _g1 = DNS[DNS.w == 0.1].set_index('kappa')
body = [f"{k:.2f} & {_dcell(_g0.loc[k])} & {_dcell(_g1.loc[k])}" for k in _g0.index]
table('stab:dense', 'Dense $\\kap$ grid near the class boundaries ($\\kap$ step 0.02; all 50 rules and both references on independent samples A and B of 10,000 games; pooled A+B estimates, s.e. in parentheses). Class (A/B shown when the samples disagree), switch round $s^\\star$ of the representative rule, $\\Delta_E$ and $\\Delta_C$. $\\dagger$, uncertain point (samples disagree or a gain within two s.e. of $\\epsilon=0.005$).',
      '$\\kap$ & \\multicolumn{4}{c}{$w=0$} & \\multicolumn{4}{c}{$w=0.1$}\\\\ & Class & $s^\\star$ & $\\Delta_E$ & $\\Delta_C$ & Class & $s^\\star$ & $\\Delta_E$ & $\\Delta_C$', body, 'lllllllll')
ORDER_T = ['stab:grids', 'stab:tol', 'stab:rep', 'stab:val', 'stab:theory', 'stab:flow', 'stab:quad', 'stab:wprof', 'stab:cond', 'stab:mech',
           'stab:place', 'stab:decomp', 'stab:reward', 'stab:oos', 'stab:game', 'stab:dense']
_si = '\n'.join('\n'.join(TABS[k]) for k in ORDER_T)
# limit every SI tabular to the text width (requires \usepackage{adjustbox})
_si = re.sub(r'(\\begin\{tabular\})', r'\\begin{adjustbox}{max width=\\textwidth}\1', _si)
_si = _si.replace('\\end{tabular}', '\\end{tabular}\\end{adjustbox}')
_si = re.sub(r'(?<![\d.])-(0\.0+)(?![\d])', r'\1', _si)   # no negative zeros
from compact_si_tables import compact_si_tables
_compact_si = compact_si_tables(_si)
open(os.path.join(OUT, 'si_tables.tex'), 'w', encoding='utf-8').write(_compact_si)
for _index, _block in enumerate(re.findall(r'\\begin\{table\}.*?\\end\{table\}', _compact_si, re.S), 1):
    _block = re.sub(r'\\begin\{table\}\[[^]]+\]', r'\\begin{table}[!htbp]', _block, count=1)
    open(os.path.join(OUT, f'si_table{_index}.tex'), 'w', encoding='utf-8').write(_block + '\n')
print('tables written')
