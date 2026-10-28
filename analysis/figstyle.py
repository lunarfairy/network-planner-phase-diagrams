"""Plot conventions used for all figures (portable; no external style dependency)."""
import matplotlib as mpl

PCOL = {'E': '#0072B2', 'S': '#E69F00', 'C': '#009E73', 'P': '#CC79A7', 'SP': '#CC79A7'}
PNAME = {'E': 'Exclusion', 'S': 'Conciliate-then-exclude', 'C': 'Conciliation throughout', 'P': 'Probation'}
GREY = '#6E6E6E'


def apply_figure_style(sizes=(8, 7, 6)):
    base, mid, small = sizes
    mpl.rcParams.update({"font.size": base, "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": mid,
                         "xtick.labelsize": small, "ytick.labelsize": small, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
                         "savefig.dpi": 300, "legend.frameon": False})


def panel_letter(ax, letter, dx=-0.14, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
