"""Deterministic accuracy rules shared by score_llm.py and score_campaign.py (factored, unchanged)."""
import re
from .verify import parse_dates, DATE_RE
from .concepts import COMPLICATIONS, OBS

NUMS = re.compile(r"(?<![\d-])(\d+(?:[.,]\d+)?)(?![\d-])")
def numbers(t):
    t2 = DATE_RE.sub(" ", t)
    return [float(x.replace(",", ".")) for x in NUMS.findall(t2)]

def has_date(t, d, tol=0):
    from datetime import date
    return any(abs((date.fromisoformat(x) - date.fromisoformat(d)).days) <= tol for _, x in parse_dates(t))

def has_num(t, v, tol):
    return any(abs(x - v) <= tol + 1e-9 for x in numbers(t))

def correct(row, X):
    a, g, k = row["answer"] or "", row["gold"], row["kind"]
    al = a.lower()
    if "non disponible" in al:
        return False
    tol = OBS.get(g.get("code", ""), {}).get("tol", max(0.05 * abs(g.get("value", 0) or 0), 0.1))
    if k == "date":
        return has_date(a, g["date"], 30)
    if k == "value_date":
        return has_num(a, g["value"], tol) and has_date(a, g["date"])
    if k == "value_date_stage":
        st = g["stage"].lower()
        stage_ok = st in al or (st[-1].isdigit() and re.search(r"stade\s*(g\s*)?" + st[-1] + r"(?![0-9ab])", al))
        return has_num(a, g["value"], tol) and has_date(a, g["date"]) and bool(stage_ok)
    if k == "trend":
        words = {w for w in ("hausse", "baisse", "stable") if w in al}
        return words == {g["trend"]}
    if k == "count":
        return g["count"] in [int(x) for x in numbers(a) if x == int(x)]
    if k == "bp":
        return has_num(a, g["sys"], 1) and has_num(a, g["dia"], 1) and has_date(a, g["date"])
    if k == "yes_date":
        if g["yes"]:
            return ("oui" in al[:20]) and has_date(a, g["date"], 30)
        return bool(re.search(r"\bnon\b|jamais|aucun", al))
    if k == "value_date_yes":
        ok = has_num(a, g["value"], tol) and has_date(a, g["date"])
        neg = bool(re.search(r"n'est pas sous|pas de statine|aucune statine|non sous", al))
        named = {c[1] for c in X.extract(a) if c[0] == "DRUG" and c[1].endswith("statine")}
        return ok and (neg != g["yes"]) and (not g["yes"] or not named or named <= set(g["items"]))
    if k == "set":
        found = set()
        for c in X.extract(a):
            if c[0] == "DRUG": found.add(c[1])
            if c[0] == "COND" and c[1] in COMPLICATIONS: found.add(COMPLICATIONS[c[1]][0])
        gold = set(g["items"])
        if not gold:
            return not found and bool(re.search(r"aucun|pas de", al))
        return found == gold
    return False

