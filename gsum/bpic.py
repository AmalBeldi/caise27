"""Instantiation of the method for process event logs: BPI Challenge 2017 (loan applications).

Method steps (Table 3 of the paper), domain-specific configuration only:
  S1 entity = application (case); facts = events (activity, timestamp, offered amount)
  S2 task   = status review of an application at a reference time inside the case
  S3 completeness statements = (activity, day) cells; the extraction log tells which may be incomplete
  S4 indicators = last offer amount, offers in 30 days, incomplete-file calls in 14 days, first offer sent,
                  current application state, offer accepted (existence); all are generic core operators
  S5 functional properties = last offer, current state; amounts exact, dates to the day
Everything else (operators, guarantees, verification) comes from gsum.core unchanged.
"""
import random
import pandas as pd
from .core import Record

# ---- S1: loading (XES via pm4py, or CSV exported from it) -----------------------------------
CASE = ["case:concept:name", "case", "Case ID", "case_id"]
ACT = ["concept:name", "activity", "Activity"]
TIME = ["time:timestamp", "timestamp", "Complete Timestamp", "time"]
def _col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)[:20]}")

KEEP = {"concept:name", "time:timestamp", "lifecycle:transition", "OfferedAmount"}

def _read_xes(path):
    """Minimal streaming XES reader: one row per event with its attributes and the trace attributes
    (prefixed 'case:'), as pm4py would return them. Used when pm4py is not installed."""
    import gzip, xml.etree.ElementTree as ET
    op = gzip.open if path.endswith(".gz") else open
    rows, stack, trace_attrs, cur = [], [], {}, None
    with op(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag.split("}")[-1]
            if ev == "start":
                stack.append(tag)
                if tag == "trace":
                    trace_attrs = {}
                elif tag == "event":
                    cur = {}
                continue
            stack.pop()
            parent = stack[-1] if stack else None
            if tag == "event":
                cur["case:concept:name"] = trace_attrs.get("concept:name")
                rows.append(cur); cur = None; el.clear()
            elif tag == "trace":
                el.clear()
            elif el.get("key") is not None:
                if parent == "event" and cur is not None and el.get("key") in KEEP:
                    cur[el.get("key")] = el.get("value")
                elif parent == "trace":
                    trace_attrs[el.get("key")] = el.get("value")
    return pd.DataFrame(rows)

def load_log(path):
    if path.endswith((".xes", ".xes.gz")):
        try:
            import pm4py
            df = pm4py.read_xes(path)
        except ImportError:                       # no pm4py: minimal streaming XES reader
            df = _read_xes(path)
    else:
        df = pd.read_csv(path)
    c, a, t = _col(df, CASE), _col(df, ACT), _col(df, TIME)
    df = df.rename(columns={c: "case", a: "activity", t: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    if "lifecycle:transition" in df.columns:          # W_ work items: keep completions only
        df = df[df["lifecycle:transition"].astype(str).str.lower().isin(["complete", "nan"])]
    amount = "OfferedAmount" if "OfferedAmount" in df.columns else None
    df["value"] = pd.to_numeric(df[amount], errors="coerce").fillna(1.0) if amount else 1.0
    return df[["case", "activity", "time", "value"]].dropna(subset=["time"]).sort_values(["case", "time"])

# ---- S2 and S4: task ----------------------------------------------------------------------------
OFFER, SENT, CALL_INC, ACCEPT = "O_Create Offer", "O_Sent (mail and online)", "W_Call incomplete files", "O_Accepted"
STATES = ("A_Create Application", "A_Submitted", "A_Concept", "A_Accepted", "A_Complete", "A_Validating",
          "A_Incomplete", "A_Pending", "A_Denied", "A_Cancelled")
TASK = [("last", OFFER), ("count", (OFFER, 30)), ("count", (CALL_INC, 14)), ("first", SENT),
        ("state", STATES), ("exists", ACCEPT)]
OUT_OF_TASK = [("exists", "W_Assess potential fraud"), ("count", ("W_Validate application", 30))]

def records(df, n_cases=2000, seed=2027):
    """One record per sampled case; the reference time is a random event in the second half of the case
    (an application reviewed while in progress); later events are the future and are dropped."""
    rng = random.Random(seed)
    cases = sorted(df["case"].unique())
    cases = rng.sample(cases, min(n_cases, len(cases)))
    out = []
    g = dict(tuple(df[df["case"].isin(set(cases))].groupby("case")))
    for c in cases:
        ev = g[c].reset_index(drop=True)
        if len(ev) < 6:
            continue
        k = rng.randint(len(ev) // 2, len(ev) - 1)
        ref = ev.loc[k, "time"]
        facts = [dict(id=f"{c}#{i}", prop=r.activity, t=r.time, value=float(r.value))
                 for i, r in enumerate(ev.itertuples()) if r.time <= ref]
        out.append(Record(entity=c, facts=facts, ref=ref))
    return out

# ---- claims from the complete record (true) and perturbations (false) -----------------------------
from .core import last, count, first, state, exists, Statements
def claims(R, rng):
    S = Statements(set())                      # complete record
    out = []                                   # (claim, truth)
    f, _, _ = last(R, S, OFFER)
    if f is not None:
        out.append((("LAST", OFFER, f["value"], f["t"]), True))
        out.append((("LAST", OFFER, round(f["value"] * rng.choice([0.6, 0.75, 1.25, 1.5])), f["t"]), False))
        older = [x for x in R.of(OFFER) if x["value"] != f["value"] and x["t"] < f["t"]]
        if older:                              # a stale value presented as the last one
            out.append((("LAST", OFFER, older[-1]["value"], None), False))
    for prop, days in ((OFFER, 30), (CALL_INC, 14), ("W_Validate application", 30)):
        n, _, _ = count(R, S, prop, days)
        out.append((("COUNT", prop, n, days), True))
        out.append((("COUNT", prop, n + rng.choice([-2, -1, 1, 2]) if n >= 2 else n + rng.choice([1, 2]), days), False))
    f, _, _ = first(R, S, SENT)
    if f is not None:
        out.append((("FIRST", SENT, f["t"]), True))
        out.append((("FIRST", SENT, f["t"] + pd.Timedelta(days=rng.choice([-20, -7, -3, 3, 7, 20]))), False))
    st, _, _ = state(R, S, STATES)
    if st is not None:
        out.append((("STATE", st), True))
        out.append((("STATE", rng.choice([s for s in STATES if s != st])), False))
    for prop in (ACCEPT, "W_Assess potential fraud"):
        e, _, _ = exists(R, S, prop)
        out.append(((("EXISTS" if e else "NOT_EXISTS"), prop), True))
        out.append(((("NOT_EXISTS" if e else "EXISTS"), prop), False))
    return out

def family(c):
    if c[0] in ("LAST", "COUNT", "FIRST", "STATE"):
        return "derived"
    return "negation" if c[0] == "NOT_EXISTS" else "existence"

# ---- incompleteness patterns for event logs (cf. Suriadi et al., 2017) ------------------------------
def degrade(R, pattern, rng, rate=0.0):
    facts = list(R.facts)
    if pattern == "random":
        drop = set(rng.sample(range(len(facts)), int(round(rate * len(facts)))))
    elif pattern == "outage":           # a 2-7-day outage within the last 30 days before the reference time
        L = rng.randint(2, 7); end = R.ref - pd.Timedelta(days=rng.randint(0, 30 - L)); start = end - pd.Timedelta(days=L)
        drop = {i for i, f in enumerate(facts) if start <= f["t"] <= end}
    elif pattern == "late":             # the last 1-7 days are not yet loaded (delayed batch import)
        cut = R.ref - pd.Timedelta(days=rng.randint(1, 7))
        drop = {i for i, f in enumerate(facts) if f["t"] > cut}
    else:
        raise ValueError(pattern)
    kept = [f for i, f in enumerate(facts) if i not in drop]
    removed = [f for i, f in enumerate(facts) if i in drop]
    return Record(R.entity, kept, R.ref), removed

def make_fake_log(n_cases=60, seed=1):
    """Small synthetic log with the BPIC 2017 schema, for tests only (never reported)."""
    rng = random.Random(seed); rows = []
    for c in range(n_cases):
        t = pd.Timestamp("2016-03-01") + pd.Timedelta(days=rng.randint(0, 200))
        seq = ["A_Create Application", "A_Submitted", "W_Handle leads", "A_Concept", "W_Complete application",
               "A_Accepted", "O_Create Offer", "O_Sent (mail and online)", "W_Call after offers", "A_Complete"]
        seq += rng.choice([["W_Call incomplete files", "A_Incomplete", "W_Call incomplete files", "O_Create Offer",
                            "O_Sent (mail and online)", "A_Validating", "W_Validate application", "O_Accepted", "A_Pending"],
                           ["W_Validate application", "A_Denied"], ["A_Cancelled"]])
        for a in seq:
            t += pd.Timedelta(hours=rng.randint(1, 120))
            rows.append(dict(case=f"Application_{c}", activity=a, time=t,
                             value=float(rng.choice([5000, 8000, 12000, 15000, 20000])) if a == "O_Create Offer" else 1.0))
    return pd.DataFrame(rows)
