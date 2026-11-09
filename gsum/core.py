"""Generic core: records, completeness statements, guarantee-preserving operators, three-valued verification.

A record is the list of facts of one entity: dict(id, prop, t, value). `t` is a pandas Timestamp.
Completeness statements are stored as the set of (prop, day) cells that may be incomplete (day = calendar
day); a cell not listed is declared complete. `complete(prop, t0, t1)` holds iff no incomplete cell of prop
falls in [t0, t1].
"""
from dataclasses import dataclass, field
import pandas as pd

SUP, SUPO, CON, UNV = "SUP", "SUP_not_guaranteed", "CON", "UNV"

@dataclass
class Record:
    entity: str
    facts: list                       # list of dict(id, prop, t, value)
    ref: pd.Timestamp                 # reference time (moment of the question)
    def of(self, prop, t0=None, t1=None):
        t1 = self.ref if t1 is None else t1
        return sorted((f for f in self.facts if f["prop"] == prop and (t0 is None or f["t"] >= t0) and f["t"] <= t1),
                      key=lambda f: (f["t"], f["id"]))

@dataclass
class Statements:
    incomplete: set = field(default_factory=set)        # {(prop, date)}
    def complete(self, prop, t0, t1):
        d0, d1 = (pd.Timestamp(t0).normalize() if t0 is not None else None), pd.Timestamp(t1).normalize()
        return not any(p == prop and (d0 is None or d >= d0) and d <= d1 for p, d in self.incomplete)
    @classmethod
    def declare(cls, removed, mode, rng=None, wrong_rate=0.5):
        cells = {(f["prop"], pd.Timestamp(f["t"]).normalize()) for f in removed}
        if mode == "naive":
            return cls(set())
        if mode == "accurate":
            return cls(cells)
        if mode == "wrong":
            return cls({c for c in sorted(cells, key=str) if rng.random() >= wrong_rate})
        raise ValueError(mode)

# ------------------------------------------------------------------ operators with guarantees
def last(R, S, prop):
    xs = R.of(prop)
    if not xs:
        return None, [], S.complete(prop, None, R.ref)
    f = xs[-1]
    return f, [f["id"]], S.complete(prop, f["t"], R.ref)

def count(R, S, prop, days):
    t0 = R.ref - pd.Timedelta(days=days)
    xs = R.of(prop, t0)
    return len(xs), [f["id"] for f in xs], S.complete(prop, t0, R.ref)

def first(R, S, prop):
    xs = R.of(prop)
    if not xs:
        return None, [], S.complete(prop, None, R.ref)
    return xs[0], [xs[0]["id"]], S.complete(prop, None, xs[0]["t"])

def state(R, S, state_props):
    xs = sorted((f for p in state_props for f in R.of(p)), key=lambda f: (f["t"], f["id"]))
    if not xs:
        return None, [], all(S.complete(p, None, R.ref) for p in state_props)
    f = xs[-1]
    return f["prop"], [f["id"]], all(S.complete(p, f["t"], R.ref) for p in state_props)

def exists(R, S, prop):
    xs = R.of(prop)
    return bool(xs), [f["id"] for f in xs[:1]], (True if xs else S.complete(prop, None, R.ref))

# ------------------------------------------------------------------ claims and verification
# claim kinds: ("LAST", prop, value, day|None) ("COUNT", prop, n, days) ("FIRST", prop, day)
#              ("STATE", state_prop) ("EXISTS", prop) ("NOT_EXISTS", prop)
def _same_day(a, b):
    return pd.Timestamp(a).normalize() == pd.Timestamp(b).normalize()

def check(c, R, S, regime, state_props=(), tol=1e-6):
    """regime: 'assumed' (completeness assumed, = V4), 'guarantees' (= V5) or 'open' (open world).
    Returns (label, witness fact ids)."""
    k = c[0]
    def conflict(g):                       # a conflict that relies on a possibly incomplete value
        return CON if regime in ("assumed", "open") or g else UNV
    def support(g):
        return SUP if regime != "guarantees" or g else SUPO
    def absent(g):                          # the claim relies on an absence
        return CON if regime == "assumed" or (regime == "guarantees" and g) else UNV
    if k == "LAST":
        _, prop, v, day = c
        f, w, g = last(R, S, prop)
        if f is None:
            return absent(g), []
        if abs(float(f["value"]) - float(v)) <= tol and (day is None or _same_day(day, f["t"])):
            return support(g), w
        if day is not None and pd.Timestamp(day).normalize() < f["t"].normalize():
            return CON, w                    # a later fact is recorded: sound
        return conflict(g), w
    if k == "COUNT":
        _, prop, n, days = c
        m, w, g = count(R, S, prop, days)
        if n == m:
            return support(g), w
        if n < m:
            return CON, w                    # the recorded count is a lower bound: sound
        return conflict(g), w
    if k == "FIRST":
        _, prop, day = c
        f, w, g = first(R, S, prop)
        if f is None:
            return absent(g), []
        if _same_day(day, f["t"]):
            return support(g), w
        if pd.Timestamp(day).normalize() > f["t"].normalize():
            return CON, w                    # an earlier fact is recorded: sound
        return conflict(g), w
    if k == "STATE":
        st, w, g = state(R, S, state_props)
        if st is None:
            return absent(g), []
        return (support(g), w) if st == c[1] else (conflict(g), w)
    if k == "EXISTS":
        e, w, g = exists(R, S, c[1])
        return (SUP, w) if e else (absent(g), [])
    if k == "NOT_EXISTS":
        e, w, g = exists(R, S, c[1])
        if e:
            return CON, w
        return SUP if regime == "assumed" or (regime == "guarantees" and g) else UNV, []
    raise ValueError(k)

def guaranteed_share(R, S, task):
    """Share of the task's summary values that are guaranteed (for RQ1/RQ5 reporting)."""
    vals = []
    for kind, arg in task:
        if kind == "last": vals.append(last(R, S, arg)[2])
        elif kind == "count": vals.append(count(R, S, *arg)[2])
        elif kind == "first": vals.append(first(R, S, arg)[2])
        elif kind == "state": vals.append(state(R, S, arg)[2])
        elif kind == "exists": vals.append(exists(R, S, arg)[2])
    return vals
