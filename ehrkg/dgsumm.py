"""Step 2: DGsumm — task- and profile-oriented summary graph GS.

Re-implementation, for the experiments, of the five operator families of
GraphSynth (display, filtrate, transformate, calculate, abstract) instantiated
for the task T = "suivi du diabete de type 2". Every summary node keeps
  - `prov`: the DG fact ids it was computed from (provenance, used by G_exp),
  - `covers`: the fact ids / derived facts it states explicitly (used to
    measure the evidence coverage of a context).
"""
import networkx as nx
from .concepts import (OBS, COMPLICATIONS, T2DM, THEMES, condition_theme, med_ingredient, ckd_stage)
from .questions import trend, years_before

PROFILES = {
    # theme weights of UserProfile-Graph (UP)
    "generaliste":    dict(glycemie=0.9, traitement=0.8, complications=0.7, renal=0.7, cardiovasculaire=1.0),
    "endocrinologue": dict(glycemie=1.0, traitement=1.0, complications=0.9, renal=0.6, cardiovasculaire=0.4),
    "uniforme":       {t: 1.0 for t in THEMES},
}

# ---- operators --------------------------------------------------------------
def display(G, kinds):
    """Selection of node kinds."""
    return [(n, d) for n, d in G.nodes(data=True) if d.get("kind") in kinds]

def filtrate(nodes, pred):
    """Filtering by a predicate on node attributes (relevance to task T)."""
    return [(n, d) for n, d in nodes if pred(d)]

def transformate_series(obs_nodes):
    """Structural aggregation: repeated measures of one concept -> one Series."""
    by = {}
    for n, d in obs_nodes:
        if isinstance(d["value"], float):
            by.setdefault(d["code"], []).append((d["date"], n, d["value"]))
    return {c: sorted(v) for c, v in by.items()}

def transformate_episodes(med_nodes, ref):
    """Consecutive prescriptions of one ingredient -> one DrugEpisode."""
    by = {}
    for n, d in med_nodes:
        ing = med_ingredient(d["label"])
        by.setdefault(ing, []).append((d["start"], n, d))
    eps = {}
    for ing, rows in by.items():
        rows.sort(key=lambda x: (x[0], x[1]))
        active = [x for x in rows if x[0] <= ref and (x[2]["stop"] is None or x[2]["stop"] >= ref)]
        eps[ing] = dict(first=rows[0][0], first_fid=rows[0][1], last=rows[-1][0], n=len(rows),
                        active=bool(active), active_fid=active[-1][1] if active else None,
                        prov=[x[1] for x in rows])
    return eps

def calculate_series(code, rows, ref):
    """Clinical indicators on a series."""
    rows = [x for x in rows if x[0] <= ref]
    if not rows:
        return None
    d3 = years_before(ref, 3); d2 = years_before(ref, 2)
    w3 = [x for x in rows if x[0] > d3]
    lab, slope = trend([x[0] for x in w3], [x[2] for x in w3])
    vals = [x[2] for x in rows]
    return dict(last=rows[-1][2], last_date=rows[-1][0], last_fid=rows[-1][1], first_date=rows[0][0],
                n=len(rows), min=min(vals), max=max(vals), n2y=sum(1 for x in rows if x[0] > d2),
                trend3y=lab, slope3y=slope, prov=[x[1] for x in rows])

def abstract_complications(cond_nodes):
    """Abstraction: complication conditions -> grouped microvascular concept."""
    groups = {}
    for n, d in cond_nodes:
        if d["code"] in COMPLICATIONS:
            lab, organ = COMPLICATIONS[d["code"]]
            groups.setdefault(organ, {}).setdefault(lab, []).append((d["start"], n))
    return {o: {l: sorted(v)[0] for l, v in g.items()} for o, g in groups.items()}

# ---- DGsumm -------------------------------------------------------------------
def dgsumm(G, profile="generaliste", min_weight=0.3):
    ref = G.graph["ref_date"]
    w = PROFILES[profile]
    keep = {t for t, x in w.items() if x >= min_weight}
    GS = nx.DiGraph(patient=G.graph["patient"], ref_date=ref, profile=profile)
    P = G.graph["patient"]
    pd_ = G.nodes[P]
    GS.add_node("patient", type="Patient", theme=None, attrs=dict(sexe=pd_["gender"], naissance=pd_["birth"],
                date_reference=ref), prov=[], covers=set())

    def add(nid, typ, theme, attrs, prov, covers):
        if theme is not None and theme not in keep:
            return
        GS.add_node(nid, type=typ, theme=theme, attrs=attrs, prov=prov, covers=set(covers))
        GS.add_edge("patient", nid, label="has" + typ)

    # observations -> series
    obs = filtrate(display(G, {"Observation"}), lambda d: d["code"] in OBS)
    for code, rows in transformate_series(obs).items():
        s = calculate_series(code, rows, ref)
        if not s:
            continue
        cov = {s["last_fid"], f"S:{code}:n2y"} | ({f"S:{code}:trend3y"} if s["trend3y"] else set())
        add(f"serie_{OBS[code]['label'].replace(' ', '_')}", "Series", OBS[code]["theme"],
            dict(code=code, analyte=OBS[code]["label"], unite=OBS[code]["unit"], **{k: v for k, v in s.items() if k != "prov"}),
            s["prov"], cov)
    # conditions -> problems (deduplicated), complications -> abstract group
    conds = filtrate(display(G, {"Condition"}), lambda d: condition_theme(d["code"], d["label"]) is not None
                     and d["start"] <= ref)
    seen = {}
    for n, d in sorted(conds, key=lambda x: (x[1]["start"], x[0])):
        if d["code"] in COMPLICATIONS:
            continue
        if d["code"] in seen:
            seen[d["code"]][1].append(n); continue
        seen[d["code"]] = (n, [n], d)
    for code, (first, prov, d) in seen.items():
        add(f"pb_{code}", "Problem", condition_theme(code, d["label"]),
            dict(code=code, libelle=d["label"].replace(" (disorder)", ""), debut=d["start"],
                 resolu=d["stop"] if d["stop"] and d["stop"] <= ref else None), prov, {first})
    cx = abstract_complications(conds)
    allfids = {v[1] for g in cx.values() for v in g.values()}
    add("complications", "ComplicationGroup", "complications",
        dict(organes={o: {l: v[0] for l, v in g.items()} for o, g in cx.items()}),
        sorted(allfids), allfids | {"CX:complete"})
    # medications -> episodes
    meds = filtrate(display(G, {"Medication"}), lambda d: med_ingredient(d["label"]) is not None)
    eps = transformate_episodes(meds, ref)
    for (ing, cls, theme), e in eps.items():
        cov = {e["first_fid"]} | ({e["active_fid"]} if e["active_fid"] else set())
        add(f"tt_{ing.replace(' ', '_').replace('/', '_')}", "DrugEpisode", theme,
            dict(principe=ing, classe=cls, debut=e["first"], dernier=e["last"], en_cours=e["active"],
                 n_prescriptions=e["n"]), e["prov"], cov)
    ad = [k for k in eps if k[1] == "antidiabetique"]
    add("tt_antidiabetiques", "TreatmentSummary", "traitement",
        dict(liste_complete=sorted(k[0] for k in ad), en_cours=sorted(k[0] for k in ad if eps[k]["active"])),
        [], {"TX:complete"})
    # calculate: derived indicators
    if "serie_DFG" in GS:
        a = GS.nodes["serie_DFG"]["attrs"]
        add("ind_mrc", "Indicator", "renal", dict(indicateur="stade MRC (KDIGO)", valeur=ckd_stage(a["last"]),
            base=f"DFG {a['last']} le {a['last_date']}"), [a["last_fid"]], set())
    if "serie_HbA1c" in GS:
        a = GS.nodes["serie_HbA1c"]["attrs"]
        add("ind_hba1c", "Indicator", "glycemie", dict(indicateur="HbA1c < 7 %", valeur="oui" if a["last"] < 7 else "non",
            base=f"HbA1c {a['last']} le {a['last_date']}"), [a["last_fid"]], set())
    if f"pb_{T2DM}" in GS:
        a = GS.nodes[f"pb_{T2DM}"]["attrs"]
        y = int(ref[:4]) - int(a["debut"][:4])
        add("ind_anciennete", "Indicator", "glycemie", dict(indicateur="anciennete du diabete (ans)", valeur=y,
            base=f"diagnostic le {a['debut']}"), [], set())
    return GS
