"""Shared helpers for the revision analyses: rule classes and the parsimony classifier.

A family row is one rule (s, k) evaluated on one sample.  Columns used: s, k, fin, meanc, cap and
the paired differences dE_*, dC_*, dbest_* (rule minus exclusion / conciliation throughout / sample
best for the same objective) with standard errors.
"""
import numpy as np
import pandas as pd

BIG = 1000
RANK = {'E': 0, 'C': 1, 'S': 2, 'P': 3, 'SP': 4}


def cls_of(s, k, T=15):
    if s == 0 or k == 0:
        return 'E'
    if s >= T - 1 and k >= BIG:
        return 'C'
    if k >= BIG:
        return 'S'
    if s >= T - 1:
        return 'P'
    return 'SP'


def classify(f, eps=0.005, obj='fin', T=15):
    """Parsimony rule: the simplest class (E<C<S<P<SP) containing a rule within eps of the sample best
    for objective `obj`; representative = best rule of that class.  Also returns the raw argmax."""
    f = f[f.planner == 'family'].copy()
    f['cls'] = [cls_of(s, k, T) for s, k in zip(f.s, f.k)]
    db = f['dbest_' + obj]
    ok = f[db >= -eps].copy()
    ok['r'] = ok.cls.map(RANK)
    ch = ok[ok.r == ok.r.min()].sort_values('dbest_' + obj, ascending=False).iloc[0]
    b = f.loc[db.idxmax()]
    ex = f[f.s == 0].iloc[0]
    cc = f[(f.s == T - 1) & (f.k >= BIG)].iloc[0]
    near = sorted(set(ok.cls))
    return dict(phase=ch.cls, s_star=int(ch.s), k_star=int(ch.k), raw_cls=b.cls, raw_s=int(b.s), raw_k=int(b.k),
                best=float(b[obj]), dE=float(-ex['dbest_' + obj]), dE_se=float(ex['dbest_' + obj + '_se']),
                dC=float(-cc['dbest_' + obj]), dC_se=float(cc['dbest_' + obj + '_se']),
                near_classes=''.join(near), n_near=int(len(ok)))


def locked(f, s, k, T=15):
    """Evaluation of a locked rule (s, k) on an independent sample: differences to E and C, all objectives."""
    f = f[f.planner == 'family']
    r = f[(f.s == s) & (f.k == k)].iloc[0]
    out = dict(s=s, k=k)
    for o in ('fin', 'meanc', 'cap'):
        out[o] = float(r[o]); out[o + '_se'] = float(r[o + '_se'])
        out['dE_' + o] = float(r['dE_' + o]); out['dE_' + o + '_se'] = float(r['dE_' + o + '_se'])
        out['dC_' + o] = float(r['dC_' + o]); out['dC_' + o + '_se'] = float(r['dC_' + o + '_se'])
    return out
