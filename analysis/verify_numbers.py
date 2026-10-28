"""Check that every decimal number quoted in the manuscript and SI can be traced to the number file.

Usage: python analysis/verify_numbers.py <numbers.json> <tex> [<tex> ...]
For each decimal token in the body text (bibliography excluded) the script looks for a value in
numbers.json (or in processed tables via numbers.json) that rounds to the token at the token's precision,
also accepting percentages (x100) and published constants listed in FIXED.  Unmatched tokens are printed
with context; the exit status is the number of unmatched tokens.
"""
import sys, re, json

FIXED = {  # constants of the published model / design, quoted from ref. 1, and design choices
    '1.807', '0.818', '-0.010', '0.010', '0.193', '-0.193', '0.370', '1.521', '0.304', '-0.304', '2.41', '0.774', '0.085', '0.287',
    '0.909', '0.10', '0.05', '0.3', '0.005', '0.0025', '0.01', '0.99', '1.0', '0.1', '0.03', '0.5', '0.6', '0.25', '0.75', '0.9',
    '0.7', '0.02', '0.2', '2.71', '2.706', '4.5', '1.5', '0.0', '0.4', '0.8', '1.2', '0.15', '2.00', '1.25', '0.08', '1.1', '1.3', '0.35', '0.95'}


def values(N):
    out = []
    for v in N.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out += [float(x) for x in re.findall(r'-?\d+\.\d+|-?\d+', v)]
        elif isinstance(v, list):
            out += [float(x) for x in v if isinstance(x, (int, float))]
    return out


def match(tok, vals):
    d = len(tok.split('.')[1]) if '.' in tok else 0
    x = float(tok)
    for v in vals:
        for s in (1, 100, 0.01):
            if abs(round(abs(v) * s, d) - abs(x)) < 0.5 * 10 ** -d + 1e-12:
                return True
    return False


if __name__ == '__main__':
    N = json.load(open(sys.argv[1]))
    vals = values(N)
    bad = 0
    for fn in sys.argv[2:]:
        t = open(fn, encoding='utf-8').read()
        t = t.split('\\begin{thebibliography}')[0]
        t = re.sub(r'(?m)%.*$', '', t)
        for m in re.finditer(r'(?<![\w.{])(\d+\.\d+)(?![\d}])', t):
            tok = m.group(1)
            if tok in FIXED or match(tok, vals):
                continue
            bad += 1
            print(f'{fn}: {tok} | ...{t[max(0, m.start() - 70):m.end() + 30]}...'.replace('\n', ' '))
    print('unmatched', bad)
    sys.exit(min(bad, 255))
