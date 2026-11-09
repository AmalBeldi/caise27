"""RQ3 without human annotation: verifiers against the exact ground truth of synthetic records.

An answer is wrong iff it is not correct w.r.t. the COMPLETE record (deterministic rules, ehrkg/scoring.py).
A verifier flags an answer when its verdict is not OK. For each verifier we report precision, recall and F1
for detecting wrong answers, the false-alarm rate on correct answers, and the residual risk (wrong answers
among those shown as OK).
  * our verifiers V4 (completeness assumed) and V5 (guarantees) on ALL campaign answers;
  * external verifiers (verifiers_external.py outputs) on the stratified sample, compared with V4/V5 on the
    same answers.
  python3 experiments/verifiers_auto.py results/campaign/*.jsonl --sample results/rq3_sample/sample_240.jsonl \
      --external results/external/ext_*.jsonl --out paper/generated
"""
import argparse, json, os, random, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.dg import build_dg
from ehrkg.dgsumm import dgsumm
from ehrkg import rdf as R
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, SUP, CON, UNV
from ehrkg.completeness import degrade, Statements, guard, SUPO
from ehrkg.concepts import generic_med_label
from ehrkg.scoring import correct

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--sample", default=None)
ap.add_argument("--external", nargs="*", default=[])
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", default="paper/generated")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = [json.loads(l) for f in a.files for l in open(f)]
rows = [r for r in rows if r.get("answer") and r.get("model") != "DRYRUN"]
REC = load_cohort(a.csv)
extra = sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION})
X = Extractor(extra_meds=[m for m in extra if len(m) > 3])

def kbs(rec):
    G = build_dg(rec); GS = dgsumm(G, "generaliste")
    tr, roots = R.rdf_graphsyn(GS); ts, _ = R.herse(tr, roots)
    return SummaryKB(ts), RecordKB(G)
CACHE = {}
def kb_for(pid, st):
    if (pid, st) not in CACHE:
        r = REC[pid]
        if st == "none":
            KS, KR = kbs(r); S = Statements(set(), r["patient"]["ref_date"])
        else:
            pat, rate = ("random", 0.2) if st == "random20" else (st, 0.0)
            D, removed = degrade(r, pat, random.Random(f"{a.seed}-{pid}-{st}"), rate)
            KS, KR = kbs(D); S = Statements.declare(removed, r["patient"]["ref_date"], "accurate")
        CACHE[(pid, st)] = (KS, KR, S)
    return CACHE[(pid, st)]

def verdict(labs):
    return "KO" if CON in labs else ("OK" if labs and all(l in (SUP, SUPO) for l in labs) else "UNVERIFIED")

ans = {}
for r in rows:
    st = r.get("incomplete", "none")
    KS, KR, S = kb_for(r["pid"], st)
    claims = X.extract(r["answer"])
    v4 = [l for _, l, _ in verify(claims, KS, KR, "V4")]
    v5 = [guard(c, l, KR, S) for c, l in zip(claims, v4)]
    KSs, KRs, _ = kb_for(r["pid"], "none")
    factual = any(l == CON for _, l, _ in verify(claims, KSs, KRs, "V4"))     # a claim false on the complete record
    key = (r["pid"], r["qid"], r["cond"], r["model"], st)
    ans[key] = dict(pid=r["pid"], setting=st, wrong=not correct(r, X), factual_error=factual, V4=verdict(v4), V5=verdict(v5))

for f in a.external:
    for x in map(json.loads, open(f)):
        k = (x["pid"], x["qid"], x["cond"], x.get("model"), x.get("incomplete", "none"))
        if k in ans:
            ans[k][x["method"]] = x["verdict"]

rng = np.random.default_rng(a.seed)
def metrics(L, v):
    L = [x for x in L if v in x]
    tp = sum(x["wrong"] and x[v] != "OK" for x in L); fp = sum((not x["wrong"]) and x[v] != "OK" for x in L)
    fn = sum(x["wrong"] and x[v] == "OK" for x in L); nc = sum(not x["wrong"] for x in L)
    P = 100 * tp / (tp + fp) if tp + fp else float("nan"); Rr = 100 * tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * Rr / (P + Rr) if (P == P and Rr == Rr and P + Rr) else float("nan")
    shown = [x for x in L if x[v] == "OK"]
    fe = [x for x in L if x["factual_error"]]
    nw = sum(x["wrong"] for x in L)
    dist = {f"{g}_{o}": 100 * sum((x["wrong"] == (g == "wrong")) and x[v] == o for x in L) / max(1, nw if g == "wrong" else nc)
            for g in ("correct", "wrong") for o in ("OK", "UNVERIFIED", "KO")}
    return dict(n=len(L), factual_detect=100 * sum(x[v] != "OK" for x in fe) / max(1, len(fe)),
                factual_KO=100 * sum(x[v] == "KO" for x in fe) / max(1, len(fe)), n_factual=len(fe), P=P, R=Rr, F1=F, false_alarm=100 * fp / nc if nc else float("nan"), **dist,
                shown_ok=100 * len(shown) / max(1, len(L)), risk_ok=100 * sum(x["wrong"] for x in shown) / max(1, len(shown)))

res = {"all": {}, "sample": {}}
for scope in ("complete", "incomplete"):
    L = [x for x in ans.values() if (x["setting"] == "none") == (scope == "complete")]
    res["all"][scope] = {v: metrics(L, v) for v in ("V4", "V5")}
if a.sample:
    keys = [(s["pid"], s["qid"], s["cond"], s.get("model"), s.get("incomplete", "none")) for s in map(json.loads, open(a.sample))]
    S_ = [ans[k] for k in keys if k in ans]
    names = sorted({k for x in S_ for k in x if k not in ("pid", "setting", "wrong", "factual_error")})
    for scope in ("complete", "incomplete"):
        L = [x for x in S_ if (x["setting"] == "none") == (scope == "complete")]
        res["sample"][scope] = {v: metrics(L, v) for v in names}
json.dump(res, open(os.path.join(a.out, "summary_verifiers_auto.json"), "w"), indent=1, default=float)
for part, d in res.items():
    for scope, dd in d.items():
        for v, m in dd.items():
            print(f"{part:6s} {scope:10s} {v:20s} n={m['n']:5d} P={m['P']:5.1f} R={m['R']:5.1f} F1={m['F1']:5.1f} "
                  f"FA={m['false_alarm']:5.1f} riskOK={m['risk_ok']:5.1f} | correct->KO {m['correct_KO']:4.1f} UNV {m['correct_UNVERIFIED']:4.1f} | wrong->KO {m['wrong_KO']:4.1f} UNV {m['wrong_UNVERIFIED']:4.1f} OK {m['wrong_OK']:4.1f} | factual({m['n_factual']}) flagged {m['factual_detect']:4.1f} KO {m['factual_KO']:4.1f}")
