"""Clinical vocabulary used by the pipeline (T2DM task).

Themes play the role of the topic graph of Topic Graph Summary: every concept
of the record is attached to at most one clinical theme. In the thesis the
themes are learnt with hLDA on clinical notes; Synthea produces no free-text
notes, so the themes are fixed here by a manual mapping on SNOMED CT / LOINC /
RxNorm codes (documented in the paper as a limitation).
"""
import re

T2DM = "44054006"

# --- observations (LOINC) -------------------------------------------------
OBS = {
    "4548-4":  dict(label="HbA1c", unit="%", theme="glycemie", tol=0.15),
    "2339-0":  dict(label="glycemie", unit="mg/dL", theme="glycemie", tol=2.0),
    "33914-3": dict(label="DFG", unit="mL/min/1.73m2", theme="renal", tol=1.0),
    "38483-4": dict(label="creatinine", unit="mg/dL", theme="renal", tol=0.05),
    "14959-1": dict(label="microalbuminurie", unit="mg/g", theme="renal", tol=1.0),
    "8480-6":  dict(label="PAS", unit="mmHg", theme="cardiovasculaire", tol=1.0),
    "8462-4":  dict(label="PAD", unit="mmHg", theme="cardiovasculaire", tol=1.0),
    "18262-6": dict(label="LDL", unit="mg/dL", theme="cardiovasculaire", tol=1.0),
    "2093-3":  dict(label="cholesterol total", unit="mg/dL", theme="cardiovasculaire", tol=1.0),
    "2085-9":  dict(label="HDL", unit="mg/dL", theme="cardiovasculaire", tol=1.0),
    "2571-8":  dict(label="triglycerides", unit="mg/dL", theme="cardiovasculaire", tol=1.0),
    "39156-5": dict(label="IMC", unit="kg/m2", theme="cardiovasculaire", tol=0.2),
}

# --- conditions (SNOMED CT) -----------------------------------------------
COMPLICATIONS = {
    "127013003": ("atteinte renale diabetique", "renale"),
    "90781000119102": ("microalbuminurie diabetique", "renale"),
    "157141000119108": ("proteinurie diabetique", "renale"),
    "368581000119106": ("neuropathie diabetique", "neurologique"),
    "1551000119108": ("retinopathie diabetique non proliferante", "retinienne"),
    "1501000119109": ("retinopathie diabetique proliferante", "retinienne"),
    "97331000119101": ("oedeme maculaire diabetique", "retinienne"),
}

def condition_theme(code, desc):
    d = desc.lower()
    if code == T2DM or "diabetes" in d or "prediabetes" in d or "hyperglycemia" in d:
        if code in COMPLICATIONS:
            return "complications"
        return "glycemie"
    if code in COMPLICATIONS:
        return "complications"
    if "kidney" in d or "renal" in d or "albuminuria" in d or "proteinuria" in d:
        return "renal"
    if any(k in d for k in ["hypertension", "hyperlipid", "hypertriglycerid", "ischemic heart",
                            "myocardial", "stroke", "metabolic syndrome", "obesity",
                            "coronary", "heart failure", "atrial"]):
        return "cardiovasculaire"
    if "retinopathy" in d or "neuropathy" in d or "macular" in d:
        return "complications"
    return None  # not relevant to the T2DM task -> removed by filtrate

# --- medications (RxNorm) -------------------------------------------------
def med_ingredient(desc):
    """abstract operator for drugs: product -> (ingredient label, class, theme)."""
    d = desc.lower()
    if "metformin" in d:
        return "metformine", "antidiabetique", "traitement"
    if "insulin isophane" in d:
        return "insuline NPH/rapide 70/30", "antidiabetique", "traitement"
    if any(k in d for k in ("glargine", "detemir", "degludec")):
        return "insuline basale", "antidiabetique", "traitement"
    if "insulin" in d:
        return "insuline rapide", "antidiabetique", "traitement"
    for k, lab in [("glipizide", "glipizide"), ("glyburide", "glibenclamide"),
                   ("liraglutide", "liraglutide"), ("semaglutide", "semaglutide"),
                   ("sitagliptin", "sitagliptine"), ("canagliflozin", "canagliflozine"),
                   ("empagliflozin", "empagliflozine"), ("dapagliflozin", "dapagliflozine"),
                   ("pioglitazone", "pioglitazone")]:
        if k in d:
            return lab, "antidiabetique", "traitement"
    for k, lab in [("simvastatin", "simvastatine"), ("atorvastatin", "atorvastatine"),
                   ("rosuvastatin", "rosuvastatine"), ("pravastatin", "pravastatine")]:
        if k in d:
            return lab, "statine", "cardiovasculaire"
    if "hydrochlorothiazide" in d and "lisinopril" in d:
        return "lisinopril/hydrochlorothiazide", "antihypertenseur", "cardiovasculaire"
    if "hydrochlorothiazide" in d and "losartan" in d:
        return "losartan/hydrochlorothiazide", "antihypertenseur", "cardiovasculaire"
    for k, lab in [("lisinopril", "lisinopril"), ("losartan", "losartan"),
                   ("amlodipine", "amlodipine"), ("hydrochlorothiazide", "hydrochlorothiazide")]:
        if k in d:
            return lab, "antihypertenseur", "cardiovasculaire"
    if "epoetin" in d:
        return "epoetine alfa", "autre", "renal"
    return None


def generic_med_label(desc):
    """Ingredient-level label for any drug (used for drugs outside the task)."""
    r = med_ingredient(desc)
    if r:
        return r[0]
    d = re.sub(r"\[.*?\]", "", desc)
    d = re.sub(r"^\d+(\.\d+)? (HR|ML|ACTUAT|DAY)\s+", "", d, flags=re.I)
    d = re.split(r"\s\d", d)[0]
    return d.strip().lower()

THEMES = ["glycemie", "traitement", "complications", "renal", "cardiovasculaire"]

def ckd_stage(egfr):
    if egfr >= 90: return "G1"
    if egfr >= 60: return "G2"
    if egfr >= 45: return "G3a"
    if egfr >= 30: return "G3b"
    if egfr >= 15: return "G4"
    return "G5"

# --- question -> concepts linker (shared by every condition) ---------------
# French keywords of the question mapped to concepts; also used for query
# expansion in the text-RAG baseline so that every condition gets the same
# entity-linking help.
LINKER = [
    (r"hba1c|h[ée]moglobine glyqu", dict(obs=["4548-4"], theme="glycemie", en="Hemoglobin A1c")),
    (r"diagnostiq|diab[èe]te de type 2", dict(cond=[T2DM], theme="glycemie", en="Diabetes mellitus type 2")),
    (r"antidiab[ée]tique|insuline", dict(medclass=["antidiabetique"], theme="traitement",
                                       en="insulin metformin")),
    (r"complication", dict(cond=list(COMPLICATIONS), theme="complications",
                           en="due to diabetes retinopathy neuropathy microalbuminuria proteinuria")),
    (r"dfg|filtration|mrc|r[ée]nal", dict(obs=["33914-3"], theme="renal", en="Glomerular filtration rate")),
    (r"ldl|cholest", dict(obs=["18262-6"], theme="cardiovasculaire", en="Cholesterol in LDL")),
    (r"statine", dict(medclass=["statine"], theme="cardiovasculaire", en="statin simvastatin atorvastatin")),
    (r"pression art[ée]rielle|tension", dict(obs=["8480-6", "8462-4"], theme="cardiovasculaire",
                                            en="Systolic Diastolic Blood Pressure")),
    # concepts outside the scope of the T2DM summary (used by out-of-scope questions)
    (r"kali[ée]mie|potassium", dict(obs=["6298-4"], theme="hors_tache", en="Potassium")),
    (r"douleur", dict(obs=["72514-3"], theme="hors_tache", en="Pain severity")),
    (r"an[ée]mie", dict(cond=["271737000"], theme="hors_tache", en="Anemia")),
    (r"h[ée]moglobine sanguine", dict(obs=["718-7"], theme="hors_tache", en="Hemoglobin in Blood")),
    (r"natr[ée]mie|sodium", dict(obs=["2947-0"], theme="hors_tache", en="Sodium")),
    (r"\bpoids\b", dict(obs=["29463-7"], theme="hors_tache", en="Body Weight")),
]

def link_question(q):
    q = q.lower()
    out = dict(obs=set(), cond=set(), medclass=set(), themes=set(), en=[])
    for pat, d in LINKER:
        if re.search(pat, q):
            out["obs"] |= set(d.get("obs", []))
            out["cond"] |= set(d.get("cond", []))
            out["medclass"] |= set(d.get("medclass", []))
            out["themes"].add(d["theme"])
            out["en"].append(d["en"])
    return out
