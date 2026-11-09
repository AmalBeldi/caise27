"""Algorithme 2 : compare les quatre variantes du vérifieur (V1 binaire, V2 monde ouvert,
V3 monde fermé local, V4 hiérarchique) sur une réponse, ou sur les réponses perturbées P0–P6
générées pour un patient.

Exemples :
  python3 scripts/demo_algo2.py --patient 38cbc124 --answer "La patiente est traitée par metformine et sitagliptine. Elle présente par ailleurs une anémie."
  python3 scripts/demo_algo2.py --patient 0 --perturb Q4
"""
import argparse, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.pipeline import prepare
from ehrkg.questions import build_questions
from ehrkg.perturb import perturbations
from ehrkg.concepts import generic_med_label
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--patient", default="0")
ap.add_argument("--answer", default=None)
ap.add_argument("--perturb", default=None, help="identifiant de question (Q1..Q10) : génère P0..P6")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
REC = load_cohort(a.csv); pids = sorted(REC)
pid = pids[int(a.patient)] if a.patient.isdigit() else next(p for p in pids if p.endswith(a.patient))
r = REC[pid]; P = prepare(r)
KS, KR = SummaryKB(P["rdf_star"]), RecordKB(P["G"])
X = Extractor(extra_meds=[m for m in sorted({generic_med_label(x) for x in r["medications"].DESCRIPTION}) if len(m) > 3])

answers = [("réponse", a.answer)] if a.answer else []
if a.perturb:
    q = next(x for x in build_questions(r) if x["qid"] == a.perturb)
    print("question :", q["question"], "\nréférence :", q["gold"])
    answers += perturbations(q, r, random.Random(a.seed))
for tag, text in answers:
    print(f"\n[{tag}] {text}")
    claims = X.extract(text)
    for V in ("V1", "V2", "V3", "V4"):
        res = verify(claims, KS, KR, V); st, sc = decide(res)
        print(f"  {V}: {st:<8} score={sc:.2f}  " + " | ".join(f"{c[0]}:{c[1]}→{l}" for c, l, _ in res))
