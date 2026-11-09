"""Scoped completeness statements, controlled incompleteness and guarantee-aware verification.

Used by RQ4 (verification under incompleteness) of the CAiSE version.

* A record r is taken as the complete record D*. `degrade` removes facts
  from it according to an incompleteness pattern and returns the recorded
  record D together with the removed facts.
* `Statements` encodes completeness statements at the granularity of a
  (property, calendar year) cell; property = ("OBS", LOINC code), ("MED",)
  or ("COND",). A cell is declared complete or not. Three declaration
  regimes: accurate (complete iff nothing was removed in the cell), naive
  (every cell complete: the assumption of the hierarchical verifier V4) and
  wrong (accurate, plus a fraction of incomplete cells wrongly declared
  complete, to stress hypothesis H3).
* `guard` turns the labels of the hierarchical verifier V4 into those of
  V5 (verification with guarantees): a label that relies on absence or on a
  non-monotone derived value (last value, count, trend, stage, current list,
  first date) is kept only if the statements guarantee it; otherwise a
  contradiction becomes UNV and a support becomes SUP° ("supported by the
  available records"). Labels that rely on the existence of a recorded fact
  are sound without completeness and are kept.
"""
import random
from .concepts import T2DM, med_ingredient
from .questions import years_before
from .verify import SUP, CON, UNV

SUPO = "supporte_non_garanti"          # SUP°: supported by a value that is not guaranteed

# ---------------------------------------------------------------- incompleteness patterns
TABLES = {"observations": "DATE", "medications": "START", "conditions": "START"}

def _years(table, row, col, ref):
    """Calendar years a removed fact covers: its date, or for a prescription
    every year of its validity period (it may make a drug 'current')."""
    y0 = int(str(row[col])[:4])
    if table == "medications":
        stop = row.STOP if isinstance(row.STOP, str) and row.STOP < ref else ref
        return list(range(y0, int(stop[:4]) + 1))
    return [y0]

def _prop(table, row):
    if table == "observations":
        return ("OBS", row.CODE)
    return ("MED",) if table == "medications" else ("COND",)

def _copy(r):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in r.items()}

def degrade(r, pattern, rng, rate=0.0):
    """Return (D, removed) where removed = list of (prop, year).

    pattern: 'none' | 'random' (rate in [0,1]) | 'gap' (6-18 months without data
    within the 3 years before the reference date: a patient followed elsewhere)
    | 'external' (every prescription of one drug active at the reference date
    is missing: a drug prescribed outside the institution)."""
    D = _copy(r)
    removed = []
    ref = r["patient"]["ref_date"]
    if pattern == "none":
        return D, removed
    if pattern == "random":
        for t, col in TABLES.items():
            df = D[t]
            if t == "conditions":       # keep the T2DM diagnosis: it defines the cohort
                cand = df.index[df.CODE != T2DM]
            else:
                cand = df.index
            k = int(round(rate * len(cand)))
            drop = rng.sample(list(cand), k) if k else []
            for i in drop:
                row = df.loc[i]
                removed += [(_prop(t, row), y) for y in _years(t, row, col, ref)]
            D[t] = df.drop(index=drop)
        return D, removed
    if pattern == "gap":
        from datetime import date, timedelta
        length = rng.randint(183, 548)
        end = date.fromisoformat(ref) - timedelta(days=rng.randint(30, 3 * 365 - length))
        start = end - timedelta(days=length)
        s, e = start.isoformat(), end.isoformat()
        for t, col in TABLES.items():
            df = D[t]
            m = (df[col] >= s) & (df[col] <= e)
            if t == "conditions":
                m &= df.CODE != T2DM
            for _, row in df[m].iterrows():
                removed += [(_prop(t, row), y) for y in _years(t, row, col, ref)]
            D[t] = df[~m]
        D["_gap"] = (s, e)
        return D, removed
    if pattern == "external":
        m = D["medications"]
        act = m[(m.START <= ref) & (m.STOP.isna() | (m.STOP >= ref))]
        ings = sorted({med_ingredient(x)[0] for x in act.DESCRIPTION if med_ingredient(x)})
        if not ings:
            return D, removed
        ing = rng.choice(ings)
        sel = m.DESCRIPTION.map(lambda x: bool(med_ingredient(x)) and med_ingredient(x)[0] == ing)
        for _, row in m[sel].iterrows():
            removed += [(("MED",), y) for y in _years("medications", row, "START", ref)]
        D["medications"] = m[~sel]
        D["_external"] = ing
        return D, removed
    raise ValueError(pattern)

# ---------------------------------------------------------------- statements
class Statements:
    def __init__(self, incomplete_cells, ref):
        self.inc = set(incomplete_cells)
        self.ref_year = int(ref[:4])
        self.props = {p for p, _ in self.inc}

    @classmethod
    def declare(cls, removed, ref, mode, rng=None, wrong_rate=0.5):
        true_inc = {(p, y) for p, y in removed}
        if mode == "naive":
            return cls(set(), ref)
        if mode == "accurate":
            return cls(true_inc, ref)
        if mode == "wrong":
            return cls({c for c in sorted(true_inc) if rng.random() >= wrong_rate}, ref)
        raise ValueError(mode)

    def complete(self, prop, y0, y1=None):
        y1 = self.ref_year if y1 is None else y1
        return not any((prop, y) in self.inc for y in range(min(y0, y1), max(y0, y1) + 1))

    def complete_upto(self, prop, y1=None):
        y1 = self.ref_year if y1 is None else y1
        return not any(p == prop and y <= y1 for p, y in self.inc)

def _y(d):
    return int(d[:4])

# ---------------------------------------------------------------- guarantee-aware labels
def guard(c, lab, KR, S):
    """V4 label -> V5 label for claim c, given the record KB of D and statements S."""
    t = c[0]
    ref = KR.ref
    def keep_con(g):
        return CON if g else UNV
    def keep_sup(g):
        return SUP if g else SUPO

    if t == "VALUE":
        _, code, v, dt, is_last = c
        prop = ("OBS", code)
        rows = [x for x in KR.obs.get(code, []) if x[0] <= ref]
        last_g = S.complete(prop, _y(rows[-1][0])) if rows else S.complete_upto(prop)
        if lab == SUP:
            return keep_sup(last_g) if is_last else SUP
        if lab == CON:
            if not rows:
                return keep_con(S.complete_upto(prop))
            if is_last:
                if dt is not None and dt < rows[-1][0]:
                    return CON                      # a later fact is recorded: sound
                return keep_con(last_g)
            if dt is not None and any(x[0] == dt for x in rows):
                return CON                          # same date, other value: functional conflict
            if dt is None:                          # undated value absent from the record
                return keep_con(S.complete_upto(prop))
            return keep_con(S.complete(prop, _y(dt), _y(dt)))
        return lab
    if t in ("DXDATE", "INSULIN_START"):
        d = c[2] if t == "DXDATE" else c[1]
        if t == "DXDATE":
            prop, rows = ("COND",), [x[0] for x in KR.conds.get(T2DM, [])]
        else:
            prop = ("MED",)
            rows = sorted(r[0] for k, v in KR.drugs.items() if "insul" in k for r in v)
        if not rows:
            return keep_con(S.complete_upto(prop)) if lab == CON else lab
        first = rows[0]
        if lab == SUP:
            return keep_sup(S.complete_upto(prop, _y(first)))
        if lab == CON:
            if d > first:
                return CON                          # an earlier fact is recorded: sound
            return keep_con(S.complete(prop, _y(d), _y(first)))
        return lab
    if t in ("TREND", "COUNT"):
        n = 3 if t == "TREND" else 2
        g = S.complete(("OBS", c[1]), _y(years_before(ref, n)))
        if lab == SUP:
            return keep_sup(g)
        if lab == CON:
            if t == "COUNT":
                w = years_before(ref, 2)
                recorded = sum(1 for x in KR.obs.get(c[1], []) if w < x[0] <= ref)
                if c[2] < recorded:
                    return CON                      # a recorded count is a lower bound: sound
            return keep_con(g)
        return lab
    if t == "STAGE":
        rows = [x for x in KR.obs.get("33914-3", []) if x[0] <= ref]
        g = S.complete(("OBS", "33914-3"), _y(rows[-1][0])) if rows else S.complete_upto(("OBS", "33914-3"))
        return keep_sup(g) if lab == SUP else (keep_con(g) if lab == CON else lab)
    if t == "DRUG":
        if lab == CON:
            _, name, current = c
            return keep_con(S.complete(("MED",), S.ref_year) if current else S.complete_upto(("MED",)))
        return lab
    if t == "COND":
        return keep_con(S.complete_upto(("COND",))) if lab == CON else lab
    if t == "NEG":
        if lab == SUP:
            cat = c[1]
            g = (S.complete(("MED",), S.ref_year) if cat in ("statine", "antidiabetique")
                 else S.complete_upto(("MED",)) if cat == "insuline" else S.complete_upto(("COND",)))
            return SUP if g else UNV
        return lab
    return lab

DERIVED = {"TREND", "COUNT", "STAGE"}
def claim_family(c):
    if c[0] in DERIVED or (c[0] == "VALUE" and c[4]) or c[0] in ("DXDATE", "INSULIN_START"):
        return "derived"          # non-monotone: depends on completeness
    if c[0] == "NEG":
        return "negation"
    return "existence"            # a recorded fact suffices

# ---------------------------------------------------------------- guarantees on the summary (Prop. 1)
def annotate(GS, S):
    """Attach to every summary node a dict `guaranteed` {attribute: bool}
    following the propagation rules (Table 2 of the paper)."""
    ref = GS.graph["ref_date"]; ry = int(ref[:4])
    MED, COND = ("MED",), ("COND",)
    for n, d in GS.nodes(data=True):
        a, typ, g = d["attrs"], d["type"], {}
        if typ == "Series":
            prop = ("OBS", a["code"])
            g["last"] = g["last_date"] = S.complete(prop, _y(a["last_date"]))
            g["n2y"] = S.complete(prop, _y(years_before(ref, 2)))
            g["trend3y"] = g["slope3y"] = S.complete(prop, _y(years_before(ref, 3)))
            g["n"] = g["min"] = g["max"] = g["first_date"] = S.complete_upto(prop)
        elif typ == "Problem":
            g["debut"] = S.complete_upto(COND, _y(a["debut"]))
        elif typ == "ComplicationGroup":
            g["organes"] = S.complete_upto(COND)
        elif typ == "DrugEpisode":
            g["en_cours"] = True if a["en_cours"] else S.complete(MED, ry, ry)
            g["debut"] = S.complete_upto(MED, _y(a["debut"]))
            g["n_prescriptions"] = g["dernier"] = S.complete_upto(MED)
        elif typ == "TreatmentSummary":
            g["en_cours"] = S.complete(MED, ry, ry)
            g["liste_complete"] = S.complete_upto(MED)
        elif typ == "Indicator":
            base = {"stade MRC (KDIGO)": "serie_DFG", "HbA1c < 7 %": "serie_HbA1c"}.get(a["indicateur"])
            if base and base in GS:
                g["valeur"] = GS.nodes[base]["attrs"] and S.complete(("OBS", GS.nodes[base]["attrs"]["code"]),
                                                                     _y(GS.nodes[base]["attrs"]["last_date"]))
            elif a["indicateur"].startswith("anciennete"):
                deb = str(a.get("base", ""))[-10:]
                g["valeur"] = S.complete_upto(COND, _y(deb)) if deb[:4].isdigit() else S.complete_upto(COND)
            else:
                g["valeur"] = S.complete_upto(COND)
        d["guaranteed"] = g
    return GS

def check_guarantees(GS_D, GS_star):
    """Prop. 1, empirically: every guaranteed attribute of the summary of D equals
    the same attribute in the summary of D*. Returns (n_guaranteed, n_total, mismatches)."""
    ng = nt = 0; bad = []
    for n, d in GS_D.nodes(data=True):
        for attr, ok in d.get("guaranteed", {}).items():
            if attr not in d["attrs"]:
                continue
            nt += 1
            if not ok:
                continue
            ng += 1
            other = GS_star.nodes[n]["attrs"].get(attr) if n in GS_star else None
            mine = d["attrs"][attr]
            same = (abs(mine - other) < 1e-9) if isinstance(mine, float) and isinstance(other, float) else mine == other
            if not same:
                bad.append((n, attr, mine, other))
    return ng, nt, bad
