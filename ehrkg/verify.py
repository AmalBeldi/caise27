"""Step 6: VerifyAndExplain — graph-guided verification of an answer.

Claim extraction is a deterministic lexicon + pattern extractor (French).
Each claim is checked against a knowledge base built from G*_RDF (summary
level) or from DG (record level), and labelled
   SUPPORTED / CONTRADICTED / UNVERIFIABLE
together with its witness triples (G_exp). Four verifier variants:
   V1  binary (original Algorithm 2): absent or different -> not supported
   V2  3 classes, open world: absent -> unverifiable
   V3  3 classes, local closed world on complete properties (drug list,
       problem list): absent -> contradicted
   V4  V3 + hierarchical lookup G*_RDF -> DG (avoids the circularity of
       verifying only against the summary)
"""
import re, unicodedata
from datetime import date
from .concepts import COMPLICATIONS, OBS, T2DM, med_ingredient, generic_med_label, ckd_stage

SUP, CON, UNV = "supporte", "contredit", "non_verifiable"

MONTHS = {m: i + 1 for i, m in enumerate(["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                                          "août", "septembre", "octobre", "novembre", "décembre"])}
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{1,2})/(\d{1,2})/(\d{4})|(\d{1,2})(?:er)? (" + "|".join(MONTHS) + r") (\d{4})")
NUM = r"(\d+(?:[.,]\d+)?)"

def parse_dates(t):
    out = []
    for m in DATE_RE.finditer(t):
        try:
            if m.group(1):
                out.append((m.start(), m.group(1)))
            elif m.group(2):
                out.append((m.start(), date(int(m.group(4)), int(m.group(3)), int(m.group(2))).isoformat()))
            else:
                out.append((m.start(), date(int(m.group(7)), MONTHS[m.group(6)], int(m.group(5))).isoformat()))
        except ValueError:
            pass
    return out

def num(s):
    return float(s.replace(",", "."))

# French labels of record conditions (problems) used in answers
COND_FR = {
    "hypertension essentielle": "59621000", "cardiopathie ischémique": "414545008",
    "hyperlipidémie": "55822004", "hypertriglycéridémie": "302870006", "syndrome métabolique": "237602007",
    "infarctus du myocarde": "22298006", "anémie": "271737000", "gingivite": "66366009",
    "sinusite virale": "444814009", "sinusite chronique": "40055000", "bronchite aiguë": "10509002",
    "pharyngite virale aiguë": "195662009", "polype du côlon": "68496003", "arthrose du genou": "239873007",
    "ostéoporose": "64859006", "trouble du sommeil": "39898005", "asthme": "195967001",
}
COND_FR["gingivite"] = "66383009"
# synonyms of complications (French and English forms seen in free answers)
for _syn, _code in [("neuropathie due au diabete", "368581000119106"), ("neuropathie diabetique", "368581000119106"),
                    ("neuropathy due to type 2 diabetes", "368581000119106"),
                    ("retinopathie diabetique non proliferante", "1551000119108"),
                    ("retinopathie diabetique non proliferative", "1551000119108"),
                    ("retinopathie diabetique proliferante", "1501000119109"),
                    ("retinopathie diabetique proliferative", "1501000119109"),
                    ("nephropathie diabetique", "127013003"), ("disorder of kidney due to diabetes", "127013003"),
                    ("microalbuminuria due to", "90781000119102"), ("proteinuria due to", "157141000119108"),
                    ("oedeme maculaire", "97331000119101")]:
    COND_FR.setdefault(_syn, _code)

# analytes outside the T2DM task (all LOINC concepts the record measures and the linker names)
ANALYTES = [(r"kali[ée]mie|potassium", "6298-4", 0.05), (r"natr[ée]mie|sodium", "2947-0", 0.5),
            (r"h[ée]moglobine sanguine|h[ée]moglobine(?! glyqu)(?!\s*a1c)", "718-7", 0.05),
            (r"\bpoids\b", "29463-7", 0.1), (r"(score de )?douleur", "72514-3", 0.0),
            (r"glyc[ée]mie", "2339-0", 2.0), (r"cr[ée]atinine", "38483-4", 0.05),
            (r"\bimc\b|indice de masse corporelle", "39156-5", 0.2),
            (r"cholest[ée]rol total", "2093-3", 1.0), (r"\bhdl\b", "2085-9", 1.0),
            (r"triglyc[ée]rides", "2571-8", 1.0)]
TOL = {code: tol for _, code, tol in ANALYTES}
def tol_of(code):
    return OBS[code]["tol"] if code in OBS else TOL.get(code, 0.05)
NUMWORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
            "neuf": 9, "dix": 10, "onze": 11, "douze": 12}
for code, (lab, organ) in COMPLICATIONS.items():
    COND_FR[lab] = code
FILTERED_COND_FR = {k: v for k, v in COND_FR.items() if v in
                    {"271737000", "66383009", "444814009", "40055000", "10509002", "195662009",
                     "68496003", "239873007", "64859006", "39898005", "195967001"}}

TREAT_FR = ["metformine", "insuline NPH/rapide 70/30", "insuline rapide", "insuline basale", "glipizide", "glibenclamide", "liraglutide",
            "semaglutide", "sitagliptine", "canagliflozine", "empagliflozine", "dapagliflozine", "pioglitazone",
            "simvastatine", "atorvastatine", "rosuvastatine", "pravastatine", "lisinopril", "losartan",
            "amlodipine", "hydrochlorothiazide", "epoetine alfa"]

def fold(t):
    """lower case without accents, same length as t (character-wise folding)."""
    return "".join(unicodedata.normalize("NFKD", c)[0] for c in t).lower()

# product names copied from the record (Synthea, English) -> ingredient labels used in answers
EN_DRUGS = [
    (r"insulin\s+isophane[^.;\n]*?insulin\s+regular[^.;\n]*?30\s*unt/ml[^,.;\n]*", "insuline NPH/rapide 70/30"),
    (r"insulin\s+(glargine|detemir|degludec)[^,.;\n]*", "insuline basale"),
    (r"insulin\s+(regular|lispro|aspart)[^,.;\n]*", "insuline rapide"),
    (r"\bmetformin\b(?!e)[^,.;\n]*", "metformine"),
    (r"\bsimvastatin\b(?!e)[^,.;\n]*", "simvastatine"), (r"\batorvastatin\b(?!e)[^,.;\n]*", "atorvastatine"),
    (r"\brosuvastatin\b(?!e)[^,.;\n]*", "rosuvastatine"), (r"\bpravastatin\b(?!e)[^,.;\n]*", "pravastatine"),
    (r"\bglipizide\b[^,.;\n]*", "glipizide"), (r"\bglyburide\b[^,.;\n]*", "glibenclamide"),
    (r"\bsitagliptin\b(?!e)[^,.;\n]*", "sitagliptine"), (r"\bliraglutide\b[^,.;\n]*", "liraglutide"),
    (r"\bsemaglutide\b[^,.;\n]*", "semaglutide"), (r"\bcanagliflozin\b(?!e)[^,.;\n]*", "canagliflozine"),
    (r"\bempagliflozin\b(?!e)[^,.;\n]*", "empagliflozine"), (r"\bdapagliflozin\b(?!e)[^,.;\n]*", "dapagliflozine"),
    (r"\bepoetin alfa\b[^,.;\n]*", "epoetine alfa"),
]
def normalize_drugs(t):
    for pat, fr in EN_DRUGS:
        t = re.sub(pat, fr, t, flags=re.I)
    return t

class Extractor:
    def __init__(self, extra_meds=()):
        self.meds = sorted(set(TREAT_FR) | set(extra_meds), key=len, reverse=True)
        self.conds = sorted(COND_FR, key=len, reverse=True)

    def _find_terms(self, t, terms):
        tl, found, taken = fold(t), [], []
        for term in terms:
            for m in re.finditer(re.escape(fold(term)), tl):
                if any(a <= m.start() < b for a, b in taken):
                    continue
                taken.append((m.start(), m.end())); found.append((m.start(), term))
        return sorted(found)

    def extract(self, text):
        C = []
        text = normalize_drugs(text)
        for sent in re.split(r"(?<=[.;])\s+", text):
            s = sent.lower(); dates = parse_dates(sent)
            # negations
            if re.search(r"n'est pas sous statine", s): C.append(("NEG", "statine"))
            if re.search(r"jamais reçu d'insuline", s): C.append(("NEG", "insuline"))
            if re.search(r"aucune complication", s): C.append(("NEG", "complication"))
            if re.search(r"aucun traitement antidiab", s): C.append(("NEG", "antidiabetique"))
            # diagnosis date
            if re.search(r"diagnostiqu|diagnostic", s) and "diab" in s and dates:
                C.append(("DXDATE", T2DM, dates[0][1]))
            # insulin start
            if re.search(r"insulin\w* .*(débuté|initi|depuis)", s) and dates and "jamais" not in s:
                C.append(("INSULIN_START", dates[0][1]))
            # trends
            m = re.search(r"hba1c.*\b(en hausse|en baisse|stable)\b", s)
            if m: C.append(("TREND", "4548-4", m.group(1).replace("en ", "")))
            elif "hba1c" in s:
                dirs = {d for pat, d in [(r"augment|hausse|progress", "hausse"), (r"diminu|baiss|r[ée]duit", "baisse"),
                                          (r"\bstab", "stable")] if re.search(pat, s)}
                if len(dirs) == 1: C.append(("TREND", "4548-4", dirs.pop()))
            # counts
            m = re.search(r"\b(\d+) dosages? d'hba1c", s)
            if m: C.append(("COUNT", "4548-4", int(m.group(1))))
            elif "hba1c" in s:
                m = re.search(r"\b(\d+|" + "|".join(NUMWORDS) + r")\s+(dosages?|mesures?|tests?)\b", s)
                if m: C.append(("COUNT", "4548-4", int(m.group(1)) if m.group(1).isdigit() else NUMWORDS[m.group(1)]))
            # CKD stage
            m = re.search(r"stade (g[1-5][ab]?)\b", s)
            if m: C.append(("STAGE", "G" + m.group(1)[1:]))
            # values
            for pat, code in [(r"hba1c[^0-9]{0,40}?" + NUM + r"\s*%", "4548-4"),
                              (r"dfg[^0-9]{0,40}?" + NUM, "33914-3"),
                              (r"ldl[^0-9]{0,40}?" + NUM, "18262-6")]:
                for m in re.finditer(pat, s):
                    d = [x for p, x in dates if p >= m.start()]
                    C.append(("VALUE", code, num(m.group(1)), d[0] if d else None, "derni" in s))
            sm = DATE_RE.sub(lambda x: " " * len(x.group(0)), s)      # dates masked, positions kept
            m = re.search(r"pression artérielle[^0-9]{0,40}?" + NUM + r"\s*/\s*" + NUM, s)
            if m:
                d = [x for p, x in dates if p >= m.start()]
                C.append(("VALUE", "8480-6", num(m.group(1)), d[0] if d else None, "derni" in s))
                C.append(("VALUE", "8462-4", num(m.group(2)), d[0] if d else None, "derni" in s))
            elif "pression art" in s or "systolique" in s or "diastolique" in s:
                d = [x for p, x in dates]
                found = False
                for word, code in (("systolique", "8480-6"), ("diastolique", "8462-4")):
                    m = re.search(word + r"[^0-9]{0,25}?" + NUM, sm) or re.search(NUM + r"\s*mm\s*\[?hg\]?\s*" + word, sm)
                    if m: C.append(("VALUE", code, num(m.group(1)), d[0] if d else None, "derni" in s)); found = True
                if not found:
                    m = re.search(r"pression artérielle[^0-9]{0,40}?" + NUM + r"\s*mm", sm)
                    if m: C.append(("VALUE", "8480-6", num(m.group(1)), d[0] if d else None, "derni" in s))
            for pat, code, _ in ANALYTES:
                for m in re.finditer(r"(?:" + pat + r")[^0-9]{0,40}?" + NUM, sm):
                    d = [x for p, x in dates if p >= m.start()]
                    C.append(("VALUE", code, num(m.group(m.lastindex)), d[0] if d else None, "derni" in s))
                    break
            # drugs
            current = bool(re.search(r"en cours|sous statine|sous |trait[ée]e? par", s))
            for _, term in self._find_terms(sent, self.meds):
                if "insuline" in term and re.search(r"insulinothérapie|insuline a été", s) and "INSULIN_START" in [c[0] for c in C]:
                    continue
                C.append(("DRUG", term, current))
            # conditions
            if "aucune" not in s:
                for _, term in self._find_terms(sent, self.conds):
                    C.append(("COND", COND_FR[term], term))
        return C

# ---------------------------------------------------------------- knowledge bases
class SummaryKB:
    """Index over G*_RDF triples (summary level)."""
    def __init__(self, triples):
        self.by_s = {}
        for s, p, o in triples:
            self.by_s.setdefault(s, []).append((p, o))
        self.series, self.drugs, self.problems, self.comps = {}, {}, {}, {}
        self.stage = None
        for s, po in self.by_s.items():
            d = {}
            for p, o in po:
                d.setdefault(p, []).append(o)
            typ = d.get("rdf:type", [None])[0]
            if typ == "ehr:Series":
                v, u, dt = _parse_measure(d["ehr:lastMeasure"][0])
                self.series[d["ehr:code"][0]] = dict(s=s, last=v, last_date=dt, n2y=d["ehr:countLast2Years"][0],
                                                     trend=d.get("ehr:trend3y_direction", [None])[0])
            elif typ == "ehr:DrugEpisode":
                self.drugs[d["ehr:ingredient"][0]] = dict(s=s, active=d["ehr:active"][0], start=d["ehr:period_start"][0],
                                                          cls=d["ehr:drugClass"][0])
            elif typ == "ehr:Problem":
                self.problems[d["ehr:code"][0]] = dict(s=s, onset=d["ehr:onset"][0])
            elif typ == "ehr:Indicator" and d["ehr:indicator"][0].startswith("stade MRC"):
                self.stage = (s, d["ehr:value"][0])
            if "ehr:onset" in d and "ehr:label" in d and typ is None:
                lab = d["ehr:label"][0]
                code = [c for c, (l, _) in COMPLICATIONS.items() if l == lab]
                if code:
                    self.comps[code[0]] = dict(s=s, onset=d["ehr:onset"][0])
        self.has_complication_group = any(o == "ehr:ComplicationGroup" for po in self.by_s.values() for p, o in po)
        self.has_treatment_summary = any(o == "ehr:TreatmentSummary" for po in self.by_s.values() for p, o in po)

    def witness(self, s):
        return [(s, p, o) for p, o in self.by_s.get(s, []) if p != "ehr:derivedFrom"]

def _parse_measure(lit):
    m = re.match(r"([\d.]+) (.*?) @ (\d{4}-\d{2}-\d{2})", lit)
    return float(m.group(1)), m.group(2), m.group(3)

class RecordKB:
    """Index over the full data graph DG (record level)."""
    def __init__(self, G):
        self.obs, self.drugs, self.conds = {}, {}, {}
        self.ref = G.graph["ref_date"]
        for n, d in G.nodes(data=True):
            k = d.get("kind")
            if k == "Observation" and isinstance(d["value"], float):
                self.obs.setdefault(d["code"], []).append((d["date"], d["value"], n))
            elif k == "Medication":
                lab = generic_med_label(d["label"])
                self.drugs.setdefault(lab, []).append((d["start"], d["stop"], n))
            elif k == "Condition":
                self.conds.setdefault(d["code"], []).append((d["start"], n))
        for v in self.obs.values(): v.sort()
        for v in self.drugs.values(): v.sort(key=lambda x: (x[0], x[2]))
        for v in self.conds.values(): v.sort()

def _close(a, b, tol):
    return abs(a - b) <= tol + 1e-9

def _days(a, b):
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)

# ---------------------------------------------------------------- claim checks
def check_summary(c, K, cwa):
    """-> (label, witness triples). Absent facts: UNV, or CON under local CWA."""
    t = c[0]
    absent = CON if cwa else UNV
    if t == "VALUE":
        _, code, v, dt, is_last = c
        s = K.series.get(code)
        if not s: return UNV, []
        w = K.witness(s["s"])
        if dt == s["last_date"] or (dt is None and is_last):
            return (SUP if _close(v, s["last"], OBS[code]["tol"]) else CON), w
        if is_last and dt and dt != s["last_date"]:
            return CON, w
        return UNV, []
    if t == "DXDATE":
        p = K.problems.get(T2DM)
        if not p: return UNV, []
        return (SUP if _days(p["onset"], c[2]) <= 30 else CON), K.witness(p["s"])
    if t == "INSULIN_START":
        ins = [d for k, d in K.drugs.items() if "insuline" in k]
        if not ins: return absent, []
        first = min(ins, key=lambda d: d["start"])
        return (SUP if _days(first["start"], c[1]) <= 30 else CON), K.witness(first["s"])
    if t == "TREND":
        s = K.series.get(c[1])
        if not s or not s["trend"]: return UNV, []
        return (SUP if s["trend"] == c[2] else CON), K.witness(s["s"])
    if t == "COUNT":
        s = K.series.get(c[1])
        if not s: return UNV, []
        return (SUP if s["n2y"] == c[2] else CON), K.witness(s["s"])
    if t == "STAGE":
        if not K.stage: return UNV, []
        return (SUP if K.stage[1] == c[1] else CON), K.witness(K.stage[0])
    if t == "DRUG":
        _, name, current = c
        d = K.drugs.get(name)
        if not d:
            return (absent if K.has_treatment_summary else UNV), []
        if current and not d["active"]:
            return CON, K.witness(d["s"])
        return SUP, K.witness(d["s"])
    if t == "COND":
        code = c[1]
        x = K.comps.get(code) or K.problems.get(code)
        if x: return SUP, K.witness(x["s"])
        return (absent if (code in COMPLICATIONS and K.has_complication_group) else UNV), []
    if t == "NEG":
        cat = c[1]
        if cat == "statine":
            act = [d for d in K.drugs.values() if d["cls"] == "statine" and d["active"]]
            return (CON, K.witness(act[0]["s"])) if act else (SUP if cwa else UNV, [])
        if cat == "insuline":
            ins = [d for k, d in K.drugs.items() if "insuline" in k]
            return (CON, K.witness(ins[0]["s"])) if ins else (SUP if cwa else UNV, [])
        if cat == "complication":
            return (CON, []) if K.comps else (SUP if cwa else UNV, [])
        if cat == "antidiabetique":
            act = [d for d in K.drugs.values() if d["cls"] == "antidiabetique" and d["active"]]
            return (CON, K.witness(act[0]["s"])) if act else (SUP if cwa else UNV, [])
    return UNV, []

def check_record(c, K, cwa=True):
    """Same semantics against DG (record level); witness = DG fact ids."""
    t = c[0]; absent = CON if cwa else UNV
    if t == "VALUE":
        _, code, v, dt, is_last = c
        rows = [x for x in K.obs.get(code, []) if x[0] <= K.ref]
        if not rows: return absent, []
        if dt is None and is_last: dt = rows[-1][0]
        same = [x for x in rows if x[0] == dt]
        if is_last and dt != rows[-1][0]: return CON, [rows[-1][2]]
        if not same: return absent, []
        return (SUP if any(_close(v, x[1], tol_of(code)) for x in same) else CON), [x[2] for x in same]
    if t == "DRUG":
        _, name, current = c
        rows = K.drugs.get(name) or [r for k, v in K.drugs.items() if k.startswith(name) for r in v]
        if not rows: return absent, []
        if current:
            act = [r for r in rows if r[0] <= K.ref and (r[1] is None or r[1] >= K.ref)]
            return (SUP, [act[-1][2]]) if act else (CON, [rows[-1][2]])
        return SUP, [rows[0][2]]
    if t == "COND":
        rows = K.conds.get(c[1])
        return (SUP, [rows[0][1]]) if rows else (absent, [])
    if t == "DXDATE":
        rows = K.conds.get(T2DM)
        return (SUP if rows and _days(rows[0][0], c[2]) <= 30 else CON), [rows[0][1]] if rows else []
    if t == "INSULIN_START":
        rows = sorted(r for k, v in K.drugs.items() if "insul" in k for r in v)
        if not rows: return absent, []
        return (SUP if _days(rows[0][0], c[1]) <= 30 else CON), [rows[0][2]]
    if t == "NEG":
        cat = c[1]
        if cat == "insuline":
            rows = [r for k, v in K.drugs.items() if "insul" in k for r in v if r[0] <= K.ref]
            return (CON, [rows[0][2]]) if rows else (SUP, [])
        if cat in ("statine", "antidiabetique"):
            act = [r for k, v in K.drugs.items() if (med_ingredient_cls(k) == cat) for r in v
                   if r[0] <= K.ref and (r[1] is None or r[1] >= K.ref)]
            return (CON, [act[0][2]]) if act else (SUP, [])
        if cat == "complication":
            rows = [v[0][1] for code, v in K.conds.items() if code in COMPLICATIONS]
            return (CON, rows[:1]) if rows else (SUP, [])
    return UNV, []

_CLS = {}
def med_ingredient_cls(label):
    """class of an ingredient-level label (french labels of concepts.med_ingredient)."""
    if not _CLS:
        for lab, cls in [("metformine", "antidiabetique"), ("insuline NPH/rapide 70/30", "antidiabetique"),
                         ("insuline rapide", "antidiabetique"), ("insuline basale", "antidiabetique"), ("simvastatine", "statine"),
                         ("atorvastatine", "statine"), ("rosuvastatine", "statine"), ("pravastatine", "statine")]:
            _CLS[lab] = cls
        for x in ["glipizide", "glibenclamide", "liraglutide", "semaglutide", "sitagliptine", "canagliflozine",
                  "empagliflozine", "dapagliflozine", "pioglitazone"]:
            _CLS[x] = "antidiabetique"
    return _CLS.get(label)

def verify(claims, KS, KR, variant):
    """Returns list of (claim, label, witness) and the global status."""
    out = []
    for c in claims:
        if variant == "V1":
            lab, w = check_summary(c, KS, cwa=False)
            lab = SUP if lab == SUP else CON     # binary: everything not supported counts as error
        elif variant == "V2":
            lab, w = check_summary(c, KS, cwa=False)
        elif variant == "V3":
            lab, w = check_summary(c, KS, cwa=True)
        else:  # V4 hierarchical: summary (local CWA) then record level
            lab, w = check_summary(c, KS, cwa=True)
            if lab != SUP:
                lab2, w2 = check_record(c, KR, cwa=True)
                if lab2 != UNV:
                    lab, w = lab2, w2
        out.append((c, lab, w))
    return out

def decide(results, th_ok=1.0, th_partial=0.5):
    if not results:
        return "KO", 0.0
    n = len(results); sup = sum(1 for r in results if r[1] == SUP)
    score = sup / n
    if any(r[1] == CON for r in results) or score < th_partial:
        return "KO", score
    if score < th_ok:
        return "PARTIEL", score
    return "OK", score
