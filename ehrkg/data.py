"""Load a Synthea CSV export and build per-patient records with stable fact IDs."""
import os, pickle
import pandas as pd
from .concepts import T2DM

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "synthea_t2dm_cohort")

def _csv(d, name):
    """name.csv or name.csv.gz"""
    p = os.path.join(d, name + ".csv")
    return p if os.path.exists(p) else p + ".gz"

def load_cohort(csv_dir=DEFAULT_CSV, cache="auto", alive_only=True):
    if cache == "auto":
        import hashlib
        h = hashlib.md5(os.path.abspath(csv_dir).encode()).hexdigest()[:10]
        os.makedirs("cache", exist_ok=True)
        cache = os.path.join("cache", f"cohort_{h}.pkl")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    cond = pd.read_csv(_csv(csv_dir, "conditions"), dtype=str)
    pats = pd.read_csv(_csv(csv_dir, "patients"), dtype=str)
    t2 = set(cond.loc[cond.CODE == T2DM, "PATIENT"])
    if alive_only:
        t2 &= set(pats.loc[pats.DEATHDATE.isna(), "Id"])
    ids = sorted(t2)
    tabs = {}
    for name, cols in [("encounters", None), ("observations", None), ("medications", None),
                       ("procedures", None), ("conditions", None)]:
        df = cond if name == "conditions" else pd.read_csv(_csv(csv_dir, name), dtype=str)
        df = df[df.PATIENT.isin(t2)].copy()
        rows = df["_row"] if "_row" in df.columns else df.index    # original row number (filtered export)
        df["fid"] = [f"{name[0].upper()}{i}" for i in rows]  # stable fact id
        tabs[name] = df
    tabs["encounters"]["fid"] = "E" + tabs["encounters"]["Id"].str[-12:]
    obs = tabs["observations"]
    obs["num"] = pd.to_numeric(obs.VALUE, errors="coerce")
    for t in ("observations", "medications", "procedures", "encounters", "conditions"):
        df = tabs[t]
        for c in ("DATE", "START", "STOP"):
            if c in df.columns:
                df[c] = df[c].str[:10]
    groups = {t: dict(tuple(tabs[t].groupby("PATIENT"))) for t in tabs}
    records = {}
    for pid in ids:
        r = {t: groups[t].get(pid, tabs[t].iloc[0:0]).reset_index(drop=True) for t in tabs}
        p = pats[pats.Id == pid].iloc[0]
        ref = r["encounters"].START.max()
        r["patient"] = dict(id=pid, birth=p.BIRTHDATE, gender=p.GENDER, ref_date=ref)
        records[pid] = r
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(records, f)
    return records
