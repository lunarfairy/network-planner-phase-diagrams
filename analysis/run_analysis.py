"""Run the local analysis chain: processed tables -> figures -> LaTeX tables -> Source Data -> optional manuscript number check.

Usage (from the package root): python analysis/run_analysis.py [--ms <manuscript dir>]
Requires data/raw/ (the outputs of the code/ scripts, already included in this package).
"""
import subprocess, sys, os, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--ms', help='Optional manuscript copy; omit for analysis-only output'); a = ap.parse_args()
if a.ms:
    a.ms = os.path.abspath(a.ms)
    for name in ('manuscript.tex', 'supplementary_information.tex'):
        if not os.path.isfile(os.path.join(a.ms, name)):
            ap.error(f'manuscript source not found: {os.path.join(a.ms, name)}')
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
py = sys.executable
steps = [
    [py, 'analysis/build_tables.py', 'data/raw', 'data/processed'],
    [py, 'analysis/make_figures.py', 'data/processed', 'figures'],
    [py, 'analysis/make_tables.py', 'data/processed', 'data/raw', a.ms or 'generated_tables'],
    [py, 'analysis/make_source_data.py', 'data/processed', 'Source_Data.xlsx'],
]
if a.ms:
    steps.append([py, 'analysis/verify_numbers.py', 'data/processed/numbers.json', os.path.join(a.ms, 'manuscript.tex'), os.path.join(a.ms, 'supplementary_information.tex')])
for s in steps:
    print('>>', ' '.join(s[1:]), flush=True)
    r = subprocess.run(s, cwd=R)
    if r.returncode:
        sys.exit(f'step failed: {s[1]} (exit {r.returncode})')
print('all steps completed')
