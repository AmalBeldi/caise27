"""Controlled hallucination injection into reference answers.

P0 unperturbed (correct)            P1 invented drug          P2 invented condition
P3 altered value                     P4 wrong date             P5 wrong derived fact (trend/count/stage)
P6 true fact absent from the summary GS (correct; tests the circularity of
   verifying against the summary only)
Surface variation: dates are rewritten in ISO, dd/mm/yyyy or 'j mois aaaa'
and decimal points as commas, at random, so the extractor is not tested on a
single surface form.
"""
import random, re
from datetime import date, timedelta
from .concepts import COMPLICATIONS, med_ingredient, generic_med_label, ckd_stage
from .verify import FILTERED_COND_FR, MONTHS
from .questions import fmt_num

AD_DRUGS = ["sitagliptine", "empagliflozine", "dapagliflozine", "liraglutide", "glipizide", "pioglitazone"]
STATINS = ["atorvastatine", "rosuvastatine", "pravastatine", "simvastatine"]
MONTHS_INV = {v: k for k, v in MONTHS.items()}

def surface(text, rng):
    def rd(m):
        d = date.fromisoformat(m.group(0)); k = rng.random()
        if k < 0.34: return m.group(0)
        if k < 0.67: return f"{d.day:02d}/{d.month:02d}/{d.year}"
        return f"{d.day} {MONTHS_INV[d.month]} {d.year}"
    text = re.sub(r"\d{4}-\d{2}-\d{2}", rd, text)
    if rng.random() < 0.5:
        text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return text

def shift(d, rng):
    days = rng.randint(90, 900) * rng.choice([-1, 1])
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()

def perturbations(q, r, rng):
    """Yield (ptype, text). q: question item with gold and references."""
    ref = q["references"][rng.randrange(len(q["references"]))]
    g = q["gold"]; qid = q["qid"]
    meds_ever = {generic_med_label(x) for x in r["medications"].DESCRIPTION}
    conds_ever = set(r["conditions"].CODE)
    out = [("P0", ref)]
    if qid == "Q4":
        cand = [x for x in AD_DRUGS if x not in meds_ever]
        x = rng.choice(cand)
        out.append(("P1", ref.replace("Aucun traitement antidiabétique n'est en cours.",
                                      f"Les traitements antidiabétiques en cours sont : {x}.") if not g["items"]
                    else ref.rstrip(".") + f", {x}."))
    if qid == "Q8":
        cand = [x for x in STATINS if x not in meds_ever]
        if cand:
            x = rng.choice(cand)
            if g["yes"]:
                out.append(("P1", re.sub(r"sous statine \(.*?\)", f"sous statine ({x})", ref)))
            else:
                out.append(("P1", ref.replace("n'est pas sous statine", f"est sous statine ({x})")))
    if qid == "Q5":
        cand = [c for c in COMPLICATIONS if c not in conds_ever]
        lab = COMPLICATIONS[rng.choice(cand)][0]
        out.append(("P2", f"Complications documentées : {lab}." if not g["items"] else ref.rstrip(".") + f", {lab}."))
    if qid in ("Q2", "Q6", "Q8", "Q9"):
        if qid == "Q9":
            d = rng.randint(15, 30) * rng.choice([-1, 1])
            out.append(("P3", ref.replace(f"{fmt_num(g['sys'])}/", f"{fmt_num(g['sys'] + d)}/", 1)))
        else:
            delta = {"Q2": (0.8, 2.0), "Q6": (10, 30), "Q8": (20, 50)}[qid]
            v = round(g["value"] + rng.uniform(*delta) * rng.choice([-1, 1]), 1)
            v = max(v, 0.5)
            out.append(("P3", ref.replace(fmt_num(g["value"]), fmt_num(v), 1)))
    if qid in ("Q1", "Q2") or (qid == "Q7" and g["yes"]):
        out.append(("P4", ref.replace(g["date"], shift(g["date"], rng))))
    if qid == "Q3":
        other = rng.choice([x for x in ["hausse", "baisse", "stable"] if x != g["trend"]])
        t = (f"Sur les trois dernières années, l'HbA1c est en {other}." if other != "stable"
             else "Sur les trois dernières années, l'HbA1c est stable.")
        out.append(("P5", t))
    if qid == "Q10":
        n = max(0, g["count"] + rng.choice([-3, -2, -1, 1, 2, 3]))
        if n == g["count"]: n += 1
        out.append(("P5", ref.replace(f"{g['count']} dosages", f"{n} dosages", 1)))
    if qid == "Q6":
        st = rng.choice([s for s in ["G1", "G2", "G3a", "G3b", "G4"] if s != g["stage"]])
        out.append(("P5", ref.replace(f"stade {g['stage']}", f"stade {st}")))
    # P6: true but filtered fact
    fc = [lab for lab, code in FILTERED_COND_FR.items() if code in conds_ever]
    fm = sorted({generic_med_label(x) for x in r["medications"].DESCRIPTION if med_ingredient(x) is None})
    fm = [m for m in fm if re.fullmatch(r"[a-z][a-z \-]{3,30}", m)]
    if fc and (rng.random() < 0.5 or not fm):
        out.append(("P6", ref + f" Le patient présente par ailleurs : {rng.choice(fc)}."))
    elif fm:
        out.append(("P6", ref + f" Le patient a par ailleurs reçu {rng.choice(fm)}."))
    return [(p, surface(t, rng)) for p, t in out]
