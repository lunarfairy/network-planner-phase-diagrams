"""Render Figs 1-5 and Supplementary Figs 1-4 and archived branch trajectories from the processed tables.

Usage: python analysis/make_figures.py <processed_dir> <figure_dir>
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_figure_style, panel_letter, PCOL, PNAME, GREY

P, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
apply_figure_style()
EPS = 0.005
CODE = {'E': 0, 'S': 1, 'C': 2, 'P': 3, 'SP': 3}
CMAP = ListedColormap([PCOL['E'], PCOL['S'], PCOL['C'], PCOL['P']])
NORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
ORDER = ['static', 'random', 'clustering', 'graphnet', 'encouragement', 'neutral', 'maxconn']
CLAB = {'static': 'Static', 'random': 'Random', 'clustering': 'Clustering', 'graphnet': 'GraphNet*', 'encouragement': 'Encouragement',
        'neutral': 'Neutral', 'maxconn': 'Max. connectivity'}
PH = pd.read_csv(os.path.join(P, 'phase_table_certified.csv'))


def edges(v):
    v = np.asarray(v, float); m = (v[1:] + v[:-1]) / 2
    return np.r_[v[0] - (m[0] - v[0]), m, v[-1] + (v[-1] - m[-1])]


def phase_mesh(ax, d, xc, yc, col='AB_phase', mark_unc=True):
    X = np.sort(d[xc].unique()); Y = np.sort(d[yc].unique())
    Z = d.assign(c=d[col].map(CODE)).pivot_table(index=yc, columns=xc, values='c', aggfunc='first').reindex(index=Y, columns=X).values
    ax.pcolormesh(edges(X), edges(Y), Z, cmap=CMAP, norm=NORM, edgecolor='white', linewidth=0.3)
    if mark_unc and 'uncertain' in d:
        u = d[d.uncertain.astype(bool)]
        ax.scatter(u[xc], u[yc], s=3, c='k', marker='o', linewidths=0, zorder=4)
    return X, Y


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf')); fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    plt.close(fig)


def legend_phases(ax, keys=('E', 'S', 'C'), loc='upper left', fontsize=6, **kw):
    h = [mpl.patches.Patch(color=PCOL[k], label=PNAME[k]) for k in keys]
    ax.legend(handles=h, loc=loc, fontsize=fontsize, handlelength=1.0, **kw)

def phase_fig_legend(fig, keys=('E', 'S', 'C', 'P')):
    h = [mpl.patches.Patch(color=PCOL[k], label=PNAME[k]) for k in keys]
    fig.legend(handles=h, loc='outside upper center', ncol=len(keys), fontsize=6.5, handlelength=1.2)



# ============================================================ Figure 1
def fig1():
    kw = PH[PH.grid == 'kw']
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    ax = axs[0, 0]
    phase_mesh(ax, kw, 'kappa', 'w')
    ax.plot([1.0], [0.0], marker='*', ms=9, mfc='white', mec='k', mew=0.8, clip_on=False, zorder=6)
    ax.annotate('$\\kappa=1$, $w=0$', xy=(1.0, 0.0), xytext=(0.25, 0.13), fontsize=6, color='white', arrowprops=dict(arrowstyle='-', lw=0.5, color='white'))
    ax.set_xlabel('Defector responsiveness $\\kappa$'); ax.set_ylabel('Payoff-imitation share $w$')
    ax.set_title('Simplest near-best rule class (dots: uncertain)', loc='left')
    h = [mpl.patches.Patch(color=PCOL[k], label=PNAME[k]) for k in ('E', 'S', 'C')] + [mpl.patches.Patch(color=PCOL['P'], label='Probation (any history rule)')]
    fig.legend(handles=h, loc='outside upper center', ncol=4, fontsize=7, handlelength=1.2)
    panel_letter(ax, 'a')
    ax = axs[0, 1]
    Z = kw.pivot(index='w', columns='kappa', values='AB_s_star')
    Ph = kw.pivot(index='w', columns='kappa', values='AB_phase')
    M = np.where(Ph.values == 'S', Z.values, np.nan)
    X, Y = Z.columns.values, Z.index.values
    ax.pcolormesh(edges(X), edges(Y), np.where(Ph.values == 'S', np.nan, 0), cmap=ListedColormap(['#EDEDED']), edgecolor='white', linewidth=0.3)
    im = ax.pcolormesh(edges(X), edges(Y), M, cmap='YlOrBr', vmin=0, vmax=14, edgecolor='white', linewidth=0.3)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label('Switch round $s^*$', fontsize=7)
    ax.set_xlabel('Defector responsiveness $\\kappa$'); ax.set_ylabel('Payoff-imitation share $w$')
    ax.set_title('Switch round $s^*$ in the conciliate-then-exclude phase', loc='left')
    panel_letter(ax, 'b')
    ax = axs[1, 0]
    w0 = kw[kw.w == 0].sort_values('kappa')
    for col, lab, c in (('AB_dE', 'gain over exclusion $\\Delta_E$', PCOL['E']), ('AB_dC', 'gain over conciliation throughout $\\Delta_C$', PCOL['C'])):
        ax.errorbar(w0.kappa, w0[col], yerr=2 * w0[col + '_se'], color=c, marker='o', ms=3, capsize=1.5, lw=1, label=lab)
    ax.axhline(EPS, color=GREY, lw=0.6, ls='--'); ax.text(1.52, EPS, '$\\epsilon$', fontsize=7, va='center', color=GREY)
    for _, r in w0.iterrows():
        ax.axvspan(r.kappa - 0.05, r.kappa + 0.05, color=PCOL[r.AB_phase], alpha=0.12, lw=0)
    ax.set_yscale('symlog', linthresh=0.01); ax.set_ylim(-0.0005, 0.5)
    ax.set_xlabel('Defector responsiveness $\\kappa$ ($w=0$)'); ax.set_ylabel('Final-round cooperation gain')
    ax.set_title('Gains of the best rule at $w=0$ ($\\pm$2 s.e.)', loc='left')
    ax.legend(loc='upper center', fontsize=6)
    panel_letter(ax, 'c')
    ax = axs[1, 1]
    rows = []
    rc = kw.AB_raw_cls.replace({'P': 'SP'}).value_counts()
    rows.append(('raw best\n(no tolerance)', {'E': rc.get('E', 0), 'S': rc.get('S', 0), 'C': rc.get('C', 0), 'P': rc.get('SP', 0)}))
    for e, col in ((0.0025, 'AB_phase_eps0.0025'), (0.005, 'AB_phase'), (0.01, 'AB_phase_eps0.01')):
        v = kw[col].value_counts(); rows.append((f'$\\epsilon$={e}', {k: v.get(k, 0) for k in ('E', 'S', 'C', 'P')}))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, cnt) in zip(y, rows):
        left = 0
        for k in ('E', 'S', 'C', 'P'):
            if cnt[k]:
                ax.barh(yi, cnt[k], left=left, color=PCOL[k], height=0.6)
                ax.text(left + cnt[k] / 2, yi, str(cnt[k]), ha='center', va='center', fontsize=6, color='white' if k != 'S' else 'k')
                left += cnt[k]
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel('Grid points of the $\\kappa$-$w$ plane (n = 208)')
    ax.set_title('Phase areas depend on the tolerance $\\epsilon$', loc='left')
    panel_letter(ax, 'd')
    save(fig, 'Fig1')


# ============================================================ Figure 2
def fig2():
    e = pd.read_csv(os.path.join(P, 'theory_endgame.csv'))
    M = pd.read_csv(os.path.join(P, 'mechanism_branches.csv'))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    ax = axs[0, 0]
    sc = ax.scatter(e.M_total, e.sim_end_diff, c=e.w, cmap='viridis', s=12, edgecolor='k', linewidths=0.2)
    lim = [min(e.M_total.min(), e.sim_end_diff.min()) - 0.005, max(e.M_total.max(), e.sim_end_diff.max()) + 0.005]
    ax.plot(lim, lim, color=GREY, lw=0.6, ls='--'); ax.axhline(0, color=GREY, lw=0.4); ax.axvline(0, color=GREY, lw=0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label('$w$', fontsize=7)
    ax.set_xlabel('First-order prediction $M_{14}$'); ax.set_ylabel('Simulated effect of the last\nconciliation step ($s=14$ vs $13$)')
    ax.set_title('Summed single-tie effects approximate the end-game decision', loc='left')
    panel_letter(ax, 'a')
    specs = [('orig_k0.8', '$\\kappa$=0.8', '#E69F00', '-'), ('orig_k1.0', '$\\kappa$=1.0', '#009E73', '-'),
             ('emp_M1w', 're-estimated, $\\hat\\kappa$', '#7B3294', '-'), ('emp_M1w_k1.0', 're-estimated, $\\kappa$=1', '#7B3294', ':'),
             ('emp_M1w_nocap', 're-estimated, no capital limit', '#7B3294', '--')]
    ax = axs[0, 1]
    for lab, name, c, ls in specs:
        g = M[M.label == lab].sort_values('s')
        ax.plot(g.s, g.imm, color=c, ls=ls, marker='o', ms=2.5, label=name)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xlabel('Planner step $s$ of the extra conciliation step'); ax.set_ylabel('Effect on cooperation\nin round $s+1$')
    ax.set_title('Immediate effect of conciliating one more step', loc='left')
    ax.legend(fontsize=5.5, loc='center right', bbox_to_anchor=(1.0, 0.45), handlelength=3.0)
    panel_letter(ax, 'b')
    ax = axs[1, 0]
    for lab, name, c, ls in specs:
        g = M[M.label == lab].sort_values('s')
        ax.errorbar(g.s, g.fin, yerr=2 * g.fin_se, color=c, ls=ls, marker='o', ms=2.5, capsize=1, lw=1, label=name)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_ylim(-0.02, 0.012)
    ax.set_xlabel('Planner step $s$'); ax.set_ylabel('Effect on final-round\ncooperation ($\\pm$2 s.e.)')
    ax.set_title('Final-round effect of the same step (matched branches)', loc='left')
    panel_letter(ax, 'c')
    ax = axs[1, 1]
    labs = [('orig_k0.8', '$\\kappa$=0.8'), ('orig_k1.0', '$\\kappa$=1.0'), ('emp_M1w', 're-estimated,\n$\\hat\\kappa$'),
            ('emp_M1w_k1.0', 're-estimated,\n$\\kappa$=1')]
    x = np.arange(len(labs)); wdt = 0.2
    for j, (col, name, c) in enumerate((('immD', 'round-$s$ defectors, round $s$+1', '#7B3294'), ('immC', 'round-$s$ cooperators, round $s$+1', '#BDBDBD'),
                                        ('finD', 'round-$s$ defectors, final round', '#C2A5CF'), ('finC', 'round-$s$ cooperators, final round', '#636363'))):
        vals = [M[(M.label == l) & (M.s <= 10)][col].mean() for l, _ in labs]
        ax.bar(x + (j - 1.5) * wdt, vals, width=wdt, color=c, label=name)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([l[1] for l in labs], fontsize=6)
    ax.set_ylabel('Contribution to population\ncooperation (mean over $s\\leq$10)')
    ax.set_title('Who gains and loses from an early step', loc='left')
    ax.legend(fontsize=5.5, loc='lower left', ncol=1)
    panel_letter(ax, 'd')
    save(fig, 'Fig2')


# ============================================================ Figure 3
def fig3():
    gp = pd.read_csv(os.path.join(P, 'game_plane.csv')); GR = pd.read_csv(os.path.join(P, 'game_plane_rules.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True, sharey=True)
    for i, (ax, w) in enumerate(zip(axs, (0.0, 0.3, 0.6))):
        d = gp[gp.w == w]
        phase_mesh(ax, d.assign(uncertain=d.A_phase != d.B_phase), 'Tg', 'Sg')
        ax.plot(2.0, -1.0, marker='*', ms=8, mfc='white', mec='k', mew=0.7, zorder=6)
        r = GR[(GR.w == w) & (GR.target == 'E') & (GR.variable == 'TS')]
        if len(r):
            t_ = r.threshold.iloc[0]; xs = np.linspace(-0.2, 3.2, 10)
            ax.plot(xs, xs - t_, color='k', lw=0.8, ls='--')
            ax.text(0.03, 0.03, f'E if $T-S\\geq${t_:g}: {r.acc_B.iloc[0]*100:.0f}% (base {r.majority_B.iloc[0]*100:.0f}%)',
                    transform=ax.transAxes, fontsize=5.5, va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.5))
        ax.set_xlim(-0.125, 3.125); ax.set_ylim(-2.125, 1.125)
        ax.set_xlabel('Temptation $T$')
        if i == 0:
            ax.set_ylabel('Sucker payoff $S$')
            h = [mpl.patches.Patch(color=PCOL[k], label=PNAME[k]) for k in ('E', 'S', 'C', 'P')]
            fig.legend(handles=h, loc='outside upper center', ncol=4, fontsize=7, handlelength=1.2)
        ax.set_title(f'$w$ = {w:g}' + (' (no imitation)' if w == 0 else ''), loc='left')
        panel_letter(ax, 'abc'[i], dx=-0.2)
    save(fig, 'Fig3')


# ============================================================ Figure 4
def fig4():
    S = pd.read_csv(os.path.join(P, 'robustness_strips.csv'))
    blocks = [('Horizon $H$', 'horizon', 'T', [5, 8, 10, 15, 20, 25, 30], lambda v: f'{int(v)}'),
              ('Disposition spread (scale of $\\sigma_\\theta$)', 'sigma', 'sd_scale', [0.25, 0.5, 1.0, 1.5, 2.0], lambda v: f'{v:g}'),
              ('Disposition mean shift', 'mu', 'mu_shift', [-1.0, -0.5, 0.0, 0.5, 1.0], lambda v: ('0' if v == 0 else f'{v:+g}')),
              ('Group size $n$ (mean degree 4.5)', 'size', 'n', [8, 16, 32, 64], lambda v: f'{int(v)}'),
              ('All acceptance probabilities $\\times\\lambda$', 'comp_all', 'lam', [0.1, 0.25, 0.5, 0.75], lambda v: f'{v:g}'),
              ('Deletion acceptance $\\times\\lambda$', 'comp_del', 'lam', [0.25, 0.5], lambda v: f'{v:g}'),
              ('Addition acceptance $\\times\\lambda$', 'comp_add', 'lam', [0.25, 0.5], lambda v: f'{v:g}'),
              ('Addition acceptance by defectors $\\varphi(+|D)$', 'phi', 'phi_AD', [0.05, 0.25, 0.45, 0.65, 0.85, 0.95], lambda v: f'{v:g}')]
    fig = plt.figure(figsize=(7.0, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    left = gs[0].subgridspec(4, 1, height_ratios=[7, 5, 5, 4]); right = gs[1].subgridspec(5, 1, height_ratios=[4, 2, 2, 6, 4])
    slots = [left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3]]
    kap = np.round(np.linspace(0, 1.5, 16), 3)
    for q, ((title, g, col, vals, fmt), sl) in enumerate(zip(blocks, slots)):
        ax = fig.add_subplot(sl)
        d = S[S.grid == g]
        Z = np.full((len(vals), len(kap)), np.nan); U = np.zeros_like(Z, bool)
        for i, v in enumerate(vals):
            for j, k in enumerate(kap):
                r = d[np.isclose(d[col], v) & np.isclose(d.kappa, k)]
                if len(r):
                    Z[i, j] = CODE[r.AB_phase.iloc[0]]; U[i, j] = bool(r.uncertain.iloc[0])
        ax.pcolormesh(edges(kap), np.arange(len(vals) + 1) - 0.5, Z, cmap=CMAP, norm=NORM, edgecolor='white', linewidth=0.3)
        yy, xx = np.nonzero(U); ax.scatter(kap[xx], yy, s=2.5, c='k', linewidths=0)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([fmt(v) for v in vals], fontsize=5.5)
        ax.set_title(title, loc='left', fontsize=7)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        if q in (3, 7):
            ax.set_xlabel('Defector responsiveness $\\kappa$')
        else:
            ax.set_xticklabels([])
        panel_letter(ax, 'abcdefgh'[q], dx=-0.16, dy=1.0)
    ax = fig.add_subplot(right[4])
    u = S[S.grid == 'unanimity'].sort_values('w')
    ax.plot(u.w, u.AB_dE, color=PCOL['E'], marker='o', ms=2.5, label='$\\Delta_E$')
    ax.plot(u.w, u.AB_dC, color=PCOL['C'], marker='o', ms=2.5, label='$\\Delta_C$')
    ax.set_ylim(0, 0.45); ax.set_xlabel('Payoff-imitation share $w$'); ax.set_ylabel('Gain of best rule')
    ax.set_title('Stylised unanimity rule: switch phase ($s^*$=10) at every $w$', loc='left', fontsize=7)
    ax.legend(fontsize=6, loc='center right')
    panel_letter(ax, 'i', dx=-0.16, dy=1.0)
    save(fig, 'Fig4')


# ============================================================ Figure 5
def fig5():
    est = pd.read_csv(os.path.join(P, 'estimates.csv')); ks = pd.read_csv(os.path.join(P, 'empirical_kslice.csv'))
    dec = pd.read_csv(os.path.join(P, 'decomposition.csv')); mo = pd.read_csv(os.path.join(P, 'empirical_multiobjective.csv'))
    oos = pd.read_csv(os.path.join(P, 'oos_table.csv')).set_index('cond').loc[ORDER]
    kw = PH[(PH.grid == 'kw') & (PH.w == 0)].sort_values('kappa')
    fig = plt.figure(figsize=(7.0, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    kc = est[(est.model == 'M1c') & est.param.str.startswith('kappa_')].copy(); kc['cond'] = kc.param.str[6:]
    kc = kc.set_index('cond').loc[ORDER]
    y = np.arange(len(ORDER))[::-1]
    ax.errorbar(kc.est, y, xerr=[kc.est - kc.lo, kc.hi - kc.est], fmt='o', ms=3.5, color='#7B3294', capsize=1.5, lw=0.9)
    pk = est[(est.model == 'M1') & (est.param == 'kappa')].iloc[0]
    ax.axvspan(pk.lo, pk.hi, color='#7B3294', alpha=0.15, lw=0); ax.axvline(pk.est, color='#7B3294', lw=0.8)
    ax.axvline(1.0, color=GREY, lw=0.8, ls='--'); ax.text(1.005, y[0] - 0.35, '$\\kappa=1$', fontsize=5.5, color=GREY, va='top')
    ax.set_yticks(y); ax.set_yticklabels([CLAB[c] for c in ORDER], fontsize=6)
    ax.set_xlabel('Defector responsiveness $\\kappa$ (95% group-bootstrap interval)')
    ax.set_title('Responsiveness by condition (band: pooled $\\hat\\kappa$)', loc='left', fontsize=7)
    panel_letter(ax, 'a')
    ax = fig.add_subplot(gs[0, 1])
    for m, ls in (('M1w', '-'), ('M3', '--')):
        g = ks[(ks.model == m) & (ks['sample'] == 'A')].sort_values('kappa')
        ax.plot(g.kappa, g.dC, color=PCOL['C'], ls=ls, marker='o', ms=2.5)
        ax.plot(g.kappa, g.dE, color=PCOL['E'], ls=ls, marker='o', ms=2.5)
    ax.plot(kw.kappa, kw.AB_dC, color=GREY, lw=0.8)
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], color=PCOL['C'], label='$\\Delta_C$ (vs conciliation throughout)'), Line2D([], [], color=PCOL['E'], label='$\\Delta_E$ (vs exclusion)'),
          Line2D([], [], color='k', ls='-', label='re-estimated M1 ($\\hat w=0$)'), Line2D([], [], color='k', ls='--', label='re-estimated M3 (round trend)'),
          Line2D([], [], color=GREY, lw=0.8, label='$\\Delta_C$ (Table 1 settings)')]
    ax.axhline(EPS, color=GREY, ls=':', lw=0.6)
    ax.set_yscale('symlog', linthresh=0.01); ax.set_ylim(-0.0005, 2.0)
    ax.set_xlabel('$\\kappa$ (other parameters fixed)'); ax.set_ylabel('Gain of best rule')
    ax.set_title('Re-estimated model: conciliation throughout never best', loc='left', fontsize=7)
    ax.legend(handles=hh, fontsize=5, loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.6)
    panel_letter(ax, 'b')
    ax = fig.add_subplot(gs[1, 0])
    items = [('decomp_SI', 'Table 1 settings'),
             ('decomp_SI_plus_kappa', '$\\kappa:1\\to0.891$'),
             ('decomp_SI_plus_sigma', '$\\sigma_\\theta:2.41\\to3.663$'),
             ('decomp_SI_plus_slopes', '$(\\beta_0,\\beta_s,\\beta_n,\\beta_r)$: settings $\\to$ estimates'),
             ('decomp2_SI_plus_b_s', '$\\beta_s:-0.193\\to-0.255$'),
             ('decomp2_SI_plus_b_n', '$\\beta_n:0.370\\to0.194$'),
             ('decomp2_SI_plus_b_r', '$\\beta_r:1.521\\to4.000$'),
             ('decomp_EMP_minus_kappa', '$\\kappa:0.891\\to1$'),
             ('decomp_EMP_minus_sigma', '$\\sigma_\\theta:3.663\\to2.41$'),
             ('decomp_EMP_minus_slopes', '$(\\beta_0,\\beta_s,\\beta_n,\\beta_r)$: estimates $\\to$ settings'),
             ('decomp2_EMP_minus_b_int_s_n', '$(\\beta_0,\\beta_s,\\beta_n)$: estimates $\\to$ settings'),
             ('decomp2_EMP_minus_b_r', '$\\beta_r:4.000\\to1.521$')]
    yv = np.arange(len(items))[::-1]
    for yi, (lab, name) in zip(yv, items):
        g = dec[dec.label == lab]; dC = g.dC.mean(); ph = g.phase.iloc[0] if g.phase.iloc[0] in PCOL else 'P'
        ax.plot([0, dC], [yi, yi], color=GREY, lw=0.8); ax.plot(dC, yi, 'o', color=PCOL[ph], ms=4.5, mec='k', mew=0.3)
    ax.axvline(EPS, color=GREY, ls=':', lw=0.6)
    ax.set_yticks(yv); ax.set_yticklabels([n for _, n in items], fontsize=5.5)
    ax.set_xlabel('Gain over conciliation throughout $\\Delta_C$ (colour = class)')
    ax.set_title('Response coefficients, not $\\kappa$ alone, move the population', loc='left', fontsize=7)
    panel_letter(ax, 'c')
    ax = fig.add_subplot(gs[1, 1])
    m = mo[mo.model == 'pooled_M1w'].set_index('rule')
    rules = [('switch_s11', 'Conciliate-then-exclude ($s$=11)', PCOL['S']), ('exclusion', 'Exclusion', PCOL['E']),
             ('conciliation', 'Conciliation throughout', PCOL['C']), ('encouragement', 'Encouragement table', '#999999')]
    metrics = [('fin', 'Final-round\ncooperation'), ('meanc', 'Mean cooperation\nover rounds'), ('cap', 'Mean final\ncapital / 10')]
    x = np.arange(len(metrics)); wdt = 0.16
    for j, (r, name, c) in enumerate(rules):
        vals = [m.loc[r, 'fin'], m.loc[r, 'meanc'], m.loc[r, 'cap'] / 10]
        ax.plot(x + (j - 1.5) * wdt, vals, 'o', color=c, ms=6, mec='k', mew=0.3, label=name)
    ax.set_ylim(0.6, 0.79); ax.set_xlim(-0.5, 2.5); ax.set_xticks(x); ax.set_xticklabels([t for _, t in metrics], fontsize=6)
    ax.set_title('At the re-estimated population the objectives disagree', loc='left', fontsize=7)
    ax.legend(fontsize=5, loc='upper right', ncol=1)
    panel_letter(ax, 'd')
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(ORDER))
    ax.errorbar(x, oos.obs, yerr=2 * oos.obs_se, fmt='s', color='k', ms=4, capsize=2, lw=0.9, label='observed (groups, $\\pm$2 s.e.)')
    ax.plot(x - 0.15, oos.SI, 'v', color=GREY, ms=5, label='Table 1 settings')
    ax.errorbar(x + 0.15, oos.loco_M3, yerr=[oos.loco_M3 - oos.loco_M3_lo, oos.loco_M3_hi - oos.loco_M3], fmt='o', color='#7B3294', ms=4,
                capsize=2, lw=0.9, label='re-estimated, condition held out (M3; 95% bootstrap)')
    ax.set_xticks(x); ax.set_xticklabels([CLAB[c] for c in ORDER], fontsize=6.5)
    ax.set_ylabel('Final-round cooperation'); ax.set_ylim(0, 1)
    ax.set_title('Held-out prediction of the seven human conditions', loc='left', fontsize=7)
    ax.legend(fontsize=5.5, loc='lower left', ncol=3)
    panel_letter(ax, 'e', dx=-0.06)
    save(fig, 'Fig5')


# ============================================================ Supplementary figures
def supp():
    th = pd.read_csv(os.path.join(P, 'theory_all_rounds.csv'))
    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    g = th[th.w == 0]
    cm = plt.get_cmap('viridis')
    for k in sorted(g.kappa.unique()):
        if k in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5):
            d = g[g.kappa == k].sort_values('t'); ax.plot(d.t, d.M_total, color=cm(k / 1.5), lw=1, label=f'{k:g}')
    ax.axhline(0, color=GREY, lw=0.5); ax.set_xlabel('Round $t$ (states generated by conciliation throughout)'); ax.set_ylabel('First-order value $M_t$')
    ax.legend(title='$\\kappa$ ($w$=0)', fontsize=5.5, title_fontsize=6, ncol=3)
    save(fig, 'SuppFig2_first_order_value')
    kw = PH[PH.grid == 'kw']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True, sharey=True)
    for ax, (col, t) in zip(axs, (('AB_phase_eps0.0025', '$\\epsilon$ = 0.0025'),
                                  ('AB_phase_eps0.01', '$\\epsilon$ = 0.01'), ('AB_raw_cls', 'raw best (no tolerance)'))):
        phase_mesh(ax, kw, 'kappa', 'w', col=col, mark_unc=False); ax.set_title(t, loc='left', fontsize=7); ax.set_xlabel('$\\kappa$')
    axs[0].set_ylabel('$w$')
    phase_fig_legend(fig)
    save(fig, 'SuppFig1_tolerance')
    RW = pd.read_csv(os.path.join(P, 'reward_objectives.csv')); RW = RW[RW.pid < 50000]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.3), constrained_layout=True, sharey=True)
    for ax, (col, t) in zip(axs, (('AB_phase_meanc', 'mean cooperation over rounds'), ('AB_phase_cap', 'mean final capital'))):
        phase_mesh(ax, kw, 'kappa', 'w', col=col, mark_unc=False); ax.set_title(t, loc='left', fontsize=7); ax.set_xlabel('$\\kappa$')
    for ax, (col, t) in zip(axs[2:], (('loss_cap', 'loss: final-capital winner'), ('loss_dutil', 'loss: original reward winner'))):
        Z = RW.pivot(index='w', columns='kappa', values=col)
        im = ax.pcolormesh(edges(Z.columns.values), edges(Z.index.values), Z.values, cmap='Greys', vmin=0, vmax=0.18, edgecolor='white', linewidth=0.3)
        ax.set_title(t, loc='left', fontsize=7); ax.set_xlabel('$\\kappa$')
    fig.colorbar(im, ax=axs[2:], shrink=0.8, label='final-round cooperation lost')
    axs[0].set_ylabel('$w$')
    phase_fig_legend(fig)
    save(fig, 'SuppFig4_objectives')
    pr = pd.read_csv(os.path.join(P, 'w_profile.csv'))
    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    for (col, t), c in zip((('mimit_t003', '$\\tau$=0.03'), ('mimit', '$\\tau$=0.1'), ('mimit_t03', '$\\tau$=0.3')), ('#0072B2', '#7B3294', '#E69F00')):
        d = pr[pr.tau_col == col].sort_values('w'); dd = 2 * (d.nll - d.nll.min())
        ax.plot(d.w.replace(0, 3e-5), dd, marker='o', ms=3, color=c, label=t)
    ax.axhline(2.706, color=GREY, ls='--', lw=0.6); ax.text(2e-2, 3.2, '95% one-sided (boundary)', fontsize=5.5, color=GREY)
    ax.set_xscale('log'); ax.set_ylim(0, 20); ax.set_xlabel('Imitation share $w$ (fixed; 0 plotted at 3e-5)'); ax.set_ylabel('Profile 2$\\Delta$NLL')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(fontsize=5.5); ax.set_title('Profile likelihood of the imitation share', loc='left', fontsize=7)
    save(fig, 'SuppFig3_imitation_profile')
    TR = pd.read_csv(os.path.join(P, 'branch_trajectories.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True, sharey=True)
    for ax, lab, t in zip(axs, ('orig_k0.8', 'orig_k1.0', 'emp_M1w'), ('published coefficients, $\\kappa$=0.8', 'published coefficients, $\\kappa$=1.0', 're-estimated population')):
        for s_, c in zip((2, 5, 8, 11), plt.get_cmap('viridis')(np.linspace(0, 0.85, 4))):
            g = TR[(TR.label == lab) & (TR.branch == s_)].sort_values('round')
            ax.plot(g['round'], g.coop_s - g.coop_sm1, color=c, marker='o', ms=2, lw=0.9, label=f's={s_}')
        ax.axhline(0, color=GREY, lw=0.5); ax.set_title(t, loc='left', fontsize=7); ax.set_xlabel('Round')
    axs[0].set_ylabel('Cooperation difference\n(rule $s$ minus rule $s-1$)'); axs[0].legend(fontsize=5.5)
    save(fig, 'Archive_branch_trajectories')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); supp()
    print('figures written to', OUT)
