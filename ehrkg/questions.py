"""Gold question/answer pairs computed directly from the raw Synthea tables.

This module is deliberately independent of the graph pipeline (it only uses
pandas on the raw tables) so that it can serve as a check of the pipeline.
Each item carries:
  - the question text (French),
  - a typed gold answer,
  - `evidence`: a list of alternative sets of fact IDs; the context of a
    condition is *sufficient* for the item if it covers at least one set,
  - one or more French reference answers (used for the verifier study).
"""
import numpy as np
import pandas as pd
from datetime import date
from .concepts import T2DM, OBS, COMPLICATIONS, med_ingredient, ckd_stage

def _d(s):
    return date.fromisoformat(s)

def years_before(ref, n):
    r = _d(ref)
    try:
        return r.replace(year=r.year - n).isoformat()
    except ValueError:
        return r.replace(year=r.year - n, day=28).isoformat()

def obs_series(r, code):
    o = r["observations"]
    o = o[(o.CODE == code) & o.num.notna()].sort_values(["DATE", "fid"])
    return o

def trend(dates, values, thr=0.1):
    """Linear slope (unit/year); 'hausse'/'baisse'/'stable'."""
    if len(values) < 3:
        return None, None
    ref = _d(dates[0])
    x = np.array([(_d(d) - ref).days / 365.25 for d in dates])
    y = np.array(values, dtype=float)
    if np.ptp(x) == 0:
        return None, None
    s = float(np.polyfit(x, y, 1)[0])
    lab = "hausse" if s > thr else ("baisse" if s < -thr else "stable")
    return lab, s

def active_meds(r, ref):
    m = r["medications"]
    m = m[(m.START <= ref) & (m.STOP.isna() | (m.STOP >= ref))]
    out = {}
    for _, row in m.sort_values(["START", "fid"]).iterrows():
        ing = med_ingredient(row.DESCRIPTION)
        if ing:
            out[ing[0]] = (ing, row)
    return out

def fmt_num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))

def build_questions(r):
    pid, ref = r["patient"]["id"], r["patient"]["ref_date"]
    Q = []
    def add(qid, text, kind, gold, evidence, refs):
        Q.append(dict(pid=pid, qid=qid, question=text, kind=kind, gold=gold,
                      evidence=[set(e) for e in evidence], references=refs))

    # Q1 diagnosis date
    c = r["conditions"]
    d = c[c.CODE == T2DM].sort_values(["START", "fid"]).iloc[0]
    add("Q1", "Quand le diabète de type 2 a-t-il été diagnostiqué ?", "date",
        dict(date=d.START), [{d.fid}],
        [f"Le diabète de type 2 a été diagnostiqué le {d.START}.",
         f"Diagnostic de diabète de type 2 posé le {d.START}."])

    # Q2 last HbA1c
    s = obs_series(r, "4548-4"); s = s[s.DATE <= ref]
    if len(s):
        last = s.iloc[-1]
        add("Q2", "Quelle est la dernière valeur d'HbA1c et à quelle date ?", "value_date",
            dict(code="4548-4", value=float(last.num), date=last.DATE), [{last.fid}],
            [f"La dernière HbA1c est de {fmt_num(last.num)} % (mesurée le {last.DATE}).",
             f"Dernier dosage d'HbA1c : {fmt_num(last.num)} % le {last.DATE}."])
        # Q3 trend over 3 years
        w = s[s.DATE > years_before(ref, 3)]
        lab, slope = trend(list(w.DATE), list(w.num))
        if lab:
            add("Q3", "Comment l'HbA1c a-t-elle évolué au cours des trois dernières années ?", "trend",
                dict(code="4548-4", trend=lab, slope=slope),
                [set(w.fid), {"S:4548-4:trend3y"}],
                [f"Sur les trois dernières années, l'HbA1c est en {lab}." if lab != "stable"
                 else "Sur les trois dernières années, l'HbA1c est stable."])
        # Q10 count over 2 years
        w2 = s[s.DATE > years_before(ref, 2)]
        add("Q10", "Combien de dosages d'HbA1c ont été réalisés au cours des deux dernières années ?",
            "count", dict(code="4548-4", count=int(len(w2))),
            [set(w2.fid), {"S:4548-4:n2y"}],
            [f"{len(w2)} dosages d'HbA1c ont été réalisés au cours des deux dernières années."])

    # Q4 current antidiabetic drugs
    act = active_meds(r, ref)
    ad = {k: v for k, v in act.items() if v[0][1] == "antidiabetique"}
    names = sorted(ad)
    ev = {v[1].fid for v in ad.values()}
    add("Q4", "Quels sont les traitements antidiabétiques en cours ?", "set",
        dict(items=names), [ev] if ev else [],
        [("Les traitements antidiabétiques en cours sont : " + ", ".join(names) + ".") if names
         else "Aucun traitement antidiabétique n'est en cours."])

    # Q5 complications
    cx = c[c.CODE.isin(COMPLICATIONS) & (c.START <= ref)].sort_values(["START", "fid"]).drop_duplicates("CODE")
    labs = sorted(COMPLICATIONS[k][0] for k in cx.CODE)
    add("Q5", "Quelles complications du diabète sont documentées ?", "set",
        dict(items=labs), [set(cx.fid)] if len(cx) else [],
        [("Complications documentées : " + ", ".join(labs) + ".") if labs
         else "Aucune complication du diabète n'est documentée."])

    # Q6 last eGFR + CKD stage
    g = obs_series(r, "33914-3"); g = g[g.DATE <= ref]
    if len(g):
        last = g.iloc[-1]; st = ckd_stage(last.num)
        add("Q6", "Quel est le dernier DFG et le stade de MRC correspondant ?", "value_date_stage",
            dict(code="33914-3", value=float(last.num), date=last.DATE, stage=st), [{last.fid}],
            [f"Le dernier DFG est de {fmt_num(last.num)} mL/min/1,73 m² ({last.DATE}), soit un stade {st}."])

    # Q7 insulin
    m = r["medications"]
    ins = m[m.DESCRIPTION.str.contains("insulin", case=False) & (m.START <= ref)].sort_values(["START", "fid"])
    if len(ins):
        add("Q7", "Le patient a-t-il déjà reçu de l'insuline ? Si oui, depuis quand ?", "yes_date",
            dict(yes=True, date=ins.iloc[0].START), [{ins.iloc[0].fid}],
            [f"Oui, une insulinothérapie a été débutée le {ins.iloc[0].START}."])
    else:
        add("Q7", "Le patient a-t-il déjà reçu de l'insuline ? Si oui, depuis quand ?", "yes_date",
            dict(yes=False), [], ["Non, le patient n'a jamais reçu d'insuline."])

    # Q8 LDL + statin
    l = obs_series(r, "18262-6"); l = l[l.DATE <= ref]
    st = {k: v for k, v in act.items() if v[0][1] == "statine"}
    if len(l):
        last = l.iloc[-1]
        ev = {last.fid} | {v[1].fid for v in st.values()}
        sname = ", ".join(sorted(st))
        add("Q8", "Quelle est la dernière valeur de LDL-cholestérol et le patient est-il sous statine ?",
            "value_date_yes", dict(code="18262-6", value=float(last.num), date=last.DATE,
                                   yes=bool(st), items=sorted(st)), [ev],
            [f"Le dernier LDL est de {fmt_num(last.num)} mg/dL ({last.DATE}) ; "
             + (f"le patient est sous statine ({sname})." if st else "le patient n'est pas sous statine.")])

    # Q9 last blood pressure
    sbp = obs_series(r, "8480-6"); sbp = sbp[sbp.DATE <= ref]
    dbp = obs_series(r, "8462-4"); dbp = dbp[dbp.DATE <= ref]
    if len(sbp) and len(dbp):
        a, b = sbp.iloc[-1], dbp[dbp.DATE == sbp.iloc[-1].DATE]
        if len(b):
            b = b.iloc[-1]
            add("Q9", "Quelle est la dernière pression artérielle mesurée ?", "bp",
                dict(sys=float(a.num), dia=float(b.num), date=a.DATE), [{a.fid, b.fid}],
                [f"La dernière pression artérielle est de {fmt_num(a.num)}/{fmt_num(b.num)} mmHg ({a.DATE})."])
    # ---- out-of-scope questions (facts removed by the T2DM summary)
    for qid, code, text in [("QX1", "6298-4", "Quelle est la dernière kaliémie mesurée ?"),
                            ("QX2", "72514-3", "Quel est le dernier score de douleur rapporté ?"),
                            ("QX4", "718-7", "Quelle est la dernière valeur d'hémoglobine sanguine ?"),
                            ("QX5", "2947-0", "Quelle est la dernière natrémie mesurée ?"),
                            ("QX6", "29463-7", "Quel est le dernier poids mesuré ?")]:
        x = obs_series(r, code); x = x[x.DATE <= ref]
        if len(x):
            last = x.iloc[-1]
            add(qid, text, "value_date", dict(code=code, value=float(last.num), date=last.DATE), [{last.fid}],
                [f"Dernière valeur : {fmt_num(last.num)} ({last.DATE})."])
    an = c[(c.CODE == "271737000") & (c.START <= ref)].sort_values(["START", "fid"])
    add("QX3", "Le patient a-t-il un antécédent d'anémie ? Si oui, depuis quand ?", "yes_date",
        dict(yes=bool(len(an)), date=an.iloc[0].START if len(an) else None),
        [{an.iloc[0].fid}] if len(an) else [], ["-"])
    return Q


def profile_keys(r):
    """Key facts a consultation summary should contain, per physician profile
    (defined a priori from the task; to be validated with clinicians)."""
    ref = r["patient"]["ref_date"]
    def last(code):
        s = obs_series(r, code); s = s[s.DATE <= ref]
        return {s.iloc[-1].fid} if len(s) else set()
    act = active_meds(r, ref)
    c = r["conditions"]
    cx = c[c.CODE.isin(COMPLICATIONS) & (c.START <= ref)].sort_values(["START", "fid"]).drop_duplicates("CODE")
    comp = set(cx.fid)
    ad = {v[1].fid for v in act.values() if v[0][1] == "antidiabetique"}
    cv = {v[1].fid for v in act.values() if v[0][1] in ("statine", "antihypertenseur")}
    probs = set()
    for code in ("59621000", "414545008"):
        x = c[(c.CODE == code) & (c.START <= ref)].sort_values(["START", "fid"])
        if len(x): probs.add(x.iloc[0].fid)
    m = r["medications"]
    ins = m[m.DESCRIPTION.str.contains("insulin", case=False) & (m.START <= ref)].sort_values(["START", "fid"])
    dx = c[c.CODE == T2DM].sort_values(["START", "fid"]).iloc[0].fid
    gp = last("4548-4") | last("33914-3") | last("8480-6") | last("8462-4") | last("18262-6") | last("39156-5") \
        | ad | cv | comp | probs
    endo = last("4548-4") | ad | comp | {dx} | last("2339-0") \
        | ({ins.iloc[0].fid} if len(ins) else set())
    return {"generaliste": gp, "endocrinologue": endo}
