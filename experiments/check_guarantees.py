"""Prop. 1 checked empirically: under every incompleteness pattern, with accurate
completeness statements, no guaranteed summary value differs from its value on the
complete record. Also reports the share of summary values that remain guaranteed.

  python3 experiments/check_guarantees.py [--seeds 3]
"""
import argparse, json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.dg import build_dg
from ehrkg.dgsumm import dgsumm
from ehrkg.completeness import degrade, Statements, annotate, check_guarantees

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--out", default="results")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=2027)
a = ap.parse_args()
REC = load_cohort(a.csv)
PATTERNS = [("random", 0.1), ("random", 0.2), ("random", 0.3), ("gap", 0.0), ("external", 0.0)]
agg = defaultdict(lambda: dict(guaranteed=0, total=0, mismatches=0, naive_mismatches=0))
examples = []
for pid in sorted(REC):
    r = REC[pid]
    GSs = dgsumm(build_dg(r), "generaliste")
    for pattern, rate in PATTERNS:
        for s in range(a.seeds):
            D, removed = degrade(r, pattern, random.Random(f"{a.seed}-{pid}-{pattern}-{rate}-{s}"), rate)
            if not removed:
                continue
            key = pattern + (str(int(100 * rate)) if pattern == "random" else "")
            GSd = dgsumm(build_dg(D), "generaliste")
            ng, nt, bad = check_guarantees(annotate(GSd, Statements.declare(removed, r["patient"]["ref_date"], "accurate")), GSs)
            ngn, _, badn = check_guarantees(annotate(dgsumm(build_dg(D), "generaliste"),
                                                     Statements.declare(removed, r["patient"]["ref_date"], "naive")), GSs)
            A = agg[key]
            A["guaranteed"] += ng; A["total"] += nt; A["mismatches"] += len(bad); A["naive_mismatches"] += len(badn)
            examples += [(key, pid[:8]) + b for b in bad[:2]]
out = {k: dict(v, share_guaranteed=100 * v["guaranteed"] / max(1, v["total"]),
               naive_wrong_share=100 * v["naive_mismatches"] / max(1, v["total"])) for k, v in agg.items()}
json.dump(out, open(f"{a.out}/prop1_check.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:9s} values={v['total']:6d} guaranteed={v['share_guaranteed']:5.1f}%  mismatches(accurate)={v['mismatches']}"
          f"  wrong values if all assumed complete={v['naive_wrong_share']:.1f}%")
if examples:
    print("examples:", examples[:10])
