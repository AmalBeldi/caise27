"""RQ4 -- verification under controlled incompleteness (no LLM needed).

For every test patient, the Synthea record is taken as the complete record D*.
Reference answers (P0, P6) and perturbed answers (P1-P5) are generated from D*
exactly as in E4. The truth of every extracted claim is its label by the
hierarchical verifier on D* (validated in E4: 100% detection, 0 false alarm):
SUP -> true, CON -> false, UNV -> excluded.

The record is then degraded into D (random loss 10/20/30%, a 6-18-month care
gap, or one externally prescribed drug), and every claim is verified on D by
  V2H  hierarchical, open world (absence -> UNV)
  V4   hierarchical, local closed world, completeness assumed (naive)
  V5a  V4 + guarantees, accurate completeness statements
  V5w  V4 + guarantees, 50% of the incomplete cells wrongly declared complete
Output: results/rq4_claims.jsonl (one row per claim x pattern x seed) and
results/rq4_summary.json.

  python3 experiments/run_incompleteness.py [--seeds 3] [--split test]
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.questions import build_questions
from ehrkg.dg import build_dg
from ehrkg.dgsumm import dgsumm
from ehrkg import rdf as R
from ehrkg.verify import (Extractor, SummaryKB, RecordKB, verify, check_summary, check_record, SUP, CON, UNV)
from ehrkg.perturb import perturbations
from ehrkg.concepts import generic_med_label
from ehrkg.completeness import degrade, Statements, guard, claim_family, SUPO

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--out", default="results")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--split", default="test", choices=["test", "val", "all"])
ap.add_argument("--seed", type=int, default=2027)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

REC = load_cohort(a.csv)
pids = sorted(REC)
val = set(random.Random(a.seed).sample(pids, len(pids) // 2))      # same split as run_symbolic.py
pids = [p for p in pids if a.split == "all" or (p in val) == (a.split == "val")]
extra = sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION})
X = Extractor(extra_meds=[m for m in extra if len(m) > 3])

PATTERNS = [("random", 0.1), ("random", 0.2), ("random", 0.3), ("gap", 0.0), ("external", 0.0)]

def kbs(r):
    G = build_dg(r); GS = dgsumm(G, "generaliste")
    tr, roots = R.rdf_graphsyn(GS); ts, _ = R.herse(tr, roots)
    return SummaryKB(ts), RecordKB(G)

def v2h(claims, KS, KR):
    out = []
    for c in claims:
        lab, _ = check_summary(c, KS, cwa=False)
        if lab != SUP:
            lab2, _ = check_record(c, KR, cwa=False)
            if lab2 != UNV:
                lab = lab2
        out.append(lab)
    return out

rows = []
t0 = time.time()
naive_mismatch = 0
for k, pid in enumerate(pids):
    r = REC[pid]
    KSs, KRs = kbs(r)
    rngp = random.Random(f"{a.seed}-{pid}")
    answers = []
    for q in build_questions(r):
        if q["qid"].startswith("QX"):
            continue
        for ptype, text in perturbations(q, r, rngp):
            claims = X.extract(text)
            truth = [lab for _, lab, _ in verify(claims, KSs, KRs, "V4")]
            answers.append((q["qid"], ptype, claims, truth))
    for pattern, rate in PATTERNS:
        for s in range(a.seeds):
            rngd = random.Random(f"{a.seed}-{pid}-{pattern}-{rate}-{s}")
            D, removed = degrade(r, pattern, rngd, rate)
            if not removed:
                continue
            KS, KR = kbs(D)
            ref = r["patient"]["ref_date"]
            S_acc = Statements.declare(removed, ref, "accurate")
            S_wr = Statements.declare(removed, ref, "wrong", rng=random.Random(f"w-{pid}-{pattern}-{rate}-{s}"))
            S_nai = Statements.declare(removed, ref, "naive")
            for ai, (qid, ptype, claims, truth) in enumerate(answers):
                v4 = [lab for _, lab, _ in verify(claims, KS, KR, "V4")]
                vo = v2h(claims, KS, KR)
                v5a = [guard(c, l, KR, S_acc) for c, l in zip(claims, v4)]
                v5w = [guard(c, l, KR, S_wr) for c, l in zip(claims, v4)]
                v5n = [guard(c, l, KR, S_nai) for c, l in zip(claims, v4)]
                naive_mismatch += sum(1 for x, y in zip(v4, v5n) if x != y)
                for j, c in enumerate(claims):
                    if truth[j] == UNV:
                        continue
                    rows.append(dict(pid=pid, pattern=pattern, rate=rate, seed=s, qid=qid, ptype=ptype, ans=ai,
                                     ctype=c[0], family=claim_family(c), truth=truth[j] == SUP,
                                     V2H=vo[j], V4=v4[j], V5a=v5a[j], V5w=v5w[j]))
    if k % 10 == 0:
        print(f"{k+1}/{len(pids)} patients  {time.time()-t0:.0f}s  rows={len(rows)}", flush=True)

assert naive_mismatch == 0, "with naive statements V5 must coincide with V4"
with open(f"{a.out}/rq4_claims.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- aggregation
VER = ["V2H", "V4", "V5a", "V5w"]
def metrics(R_):
    tr = [x for x in R_ if x["truth"]]; fa = [x for x in R_ if not x["truth"]]
    out = dict(n=len(R_), n_true=len(tr), n_false=len(fa))
    for v in VER:
        out[v] = dict(
            false_con=100 * sum(x[v] == CON for x in tr) / max(1, len(tr)),        # true claim contradicted
            false_sup=100 * sum(x[v] == SUP for x in fa) / max(1, len(fa)),        # false claim fully supported
            false_supo=100 * sum(x[v] == SUPO for x in fa) / max(1, len(fa)),      # false claim, supported by non-guaranteed value
            detect=100 * sum(x[v] == CON for x in fa) / max(1, len(fa)),           # false claim contradicted
            true_sup=100 * sum(x[v] in (SUP, SUPO) for x in tr) / max(1, len(tr)),
            unv=100 * sum(x[v] == UNV for x in R_) / max(1, len(R_)),
            decided_correct=100 * sum((x[v] in (SUP, SUPO) and x["truth"]) or (x[v] == CON and not x["truth"])
                                      for x in R_) / max(1, len(R_)))
    return out

summary = {}
groups = defaultdict(list)
for x in rows:
    key = f"{x['pattern']}{'' if x['pattern'] != 'random' else int(100 * x['rate'])}"
    groups[key].append(x)
    groups[key + "|" + x["family"]].append(x)
for kk, R_ in sorted(groups.items()):
    summary[kk] = metrics(R_)
summary["_meta"] = dict(patients=len(pids), seeds=a.seeds, split=a.split, rows=len(rows))
json.dump(summary, open(f"{a.out}/rq4_summary.json", "w"), indent=1)
print("done", time.time() - t0, "s;", len(rows), "claim rows")
