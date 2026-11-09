"""RQ5 -- second instantiation of the method on a process event log (BPI Challenge 2017).

  python3 experiments/run_rq5_bpic.py --log data/bpic2017/BPI_Challenge_2017.xes.gz --n-cases 2000
  python3 experiments/run_rq5_bpic.py --fake          # plumbing test on a synthetic log (never reported)

Reports
  * instantiation effort: lines of domain configuration (gsum/bpic.py) vs reused generic core (gsum/core.py),
    operators reused unchanged, new indicator functions, kinds of completeness statements;
  * compactness (as RQ1): events per case vs values of the task summary, tokens of both serialisations;
  * verification under controlled incompleteness (as RQ4): random loss 10/20/30%, a 2-7-day outage,
    a late batch import of the last 1-7 days; regimes: completeness assumed, guarantees with accurate
    statements, guarantees with 50% wrong statements, open world. Truth = label on the complete case.
Writes results/rq5_bpic.json and paper/generated/tab_rq5.tex.
"""
import argparse, inspect, json, os, random, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsum import core, bpic
from gsum.core import Statements, check, SUP, SUPO, CON, UNV, guaranteed_share
from ehrkg.context import ntok

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=None)
ap.add_argument("--fake", action="store_true")
ap.add_argument("--n-cases", type=int, default=2000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", default="results")
ap.add_argument("--gen", default="paper/generated")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs(a.gen, exist_ok=True)

df = bpic.make_fake_log() if a.fake else bpic.load_log(a.log)
recs = bpic.records(df, a.n_cases, a.seed)
print(f"{df['case'].nunique()} cases, {len(df)} events in the log; {len(recs)} sampled records", flush=True)

# ---- effort ------------------------------------------------------------------------------------
def loc(mod):
    src = inspect.getsource(mod).splitlines()
    return sum(1 for l in src if l.strip() and not l.strip().startswith("#"))
cfg_src = inspect.getsource(bpic)
domain_loc = sum(1 for l in cfg_src.split("def make_fake_log")[0].splitlines() if l.strip() and not l.strip().startswith("#"))
effort = dict(core_loc=loc(core), domain_loc=domain_loc,
              operators_reused=["last", "count", "first", "state", "exists"], new_indicator_functions=0,
              task_values=len(bpic.TASK), statement_kind="(activity, day) cells from the extraction/import log",
              claim_kinds=sorted({c[0] for R in recs[:50] for c, _ in bpic.claims(R, random.Random(0))}))

# ---- compactness -----------------------------------------------------------------------------------
ev_tok, sum_tok, n_ev = [], [], []
for R in recs:
    n_ev.append(len(R.facts))
    ev_tok.append(sum(ntok(f"{f['t']:%Y-%m-%d %H:%M} | {f['prop']}" + (f" | {f['value']:.0f}" if f['prop'] == bpic.OFFER else "")) for f in R.facts))
    S = Statements(set())
    lines = []
    f, _, _ = core.last(R, S, bpic.OFFER)
    lines.append(f"dernière offre : {f['value']:.0f} EUR le {f['t']:%Y-%m-%d}" if f else "aucune offre")
    for prop, d in ((bpic.OFFER, 30), (bpic.CALL_INC, 14)):
        lines.append(f"{prop} sur {d} jours : {core.count(R, S, prop, d)[0]}")
    f, _, _ = core.first(R, S, bpic.SENT); lines.append(f"première offre envoyée : {f['t']:%Y-%m-%d}" if f else "aucune offre envoyée")
    lines.append(f"état courant : {core.state(R, S, bpic.STATES)[0]}")
    lines.append(f"offre acceptée : {'oui' if core.exists(R, S, bpic.ACCEPT)[0] else 'non'}")
    sum_tok.append(sum(ntok(l) for l in lines))
compact = dict(events_median=float(np.median(n_ev)), events_tokens_median=float(np.median(ev_tok)),
               summary_values=len(bpic.TASK), summary_tokens_median=float(np.median(sum_tok)))

# ---- controlled incompleteness ---------------------------------------------------------------------
PATTERNS = [("random", 0.1), ("random", 0.2), ("random", 0.3), ("outage", 0.0), ("late", 0.0)]
REG = ["assumed", "guar_acc", "guar_wrong", "open"]
rows = []; gshare = defaultdict(list)
for R in recs:
    rngc = random.Random(f"{a.seed}-{R.entity}")
    C = bpic.claims(R, rngc)
    S0 = Statements(set())
    truth = [check(c, R, S0, "assumed", bpic.STATES)[0] for c, _ in C]   # complete record: exact labels
    for (pattern, rate) in PATTERNS:
        for s in range(a.seeds):
            rng = random.Random(f"{a.seed}-{R.entity}-{pattern}-{rate}-{s}")
            D, removed = bpic.degrade(R, pattern, rng, rate)
            if not removed:
                continue
            key = pattern + (str(int(100 * rate)) if pattern == "random" else "")
            Sacc = Statements.declare(removed, "accurate")
            Swr = Statements.declare(removed, "wrong", rng=random.Random(f"w-{R.entity}-{key}-{s}"))
            gshare[key] += guaranteed_share(D, Sacc, bpic.TASK)
            for (c, _), t in zip(C, truth):
                if t == UNV:
                    continue
                lab = dict(assumed=check(c, D, Statements(set()), "assumed", bpic.STATES)[0],
                           guar_acc=check(c, D, Sacc, "guarantees", bpic.STATES)[0],
                           guar_wrong=check(c, D, Swr, "guarantees", bpic.STATES)[0],
                           open=check(c, D, Statements(set()), "open", bpic.STATES)[0])
                rows.append(dict(case=R.entity, key=key, truth=(t == SUP), family=bpic.family(c), **lab))

rng = np.random.default_rng(a.seed)
def metrics(L, v):
    tr = [x for x in L if x["truth"]]; fa = [x for x in L if not x["truth"]]
    con_t = sum(x[v] == CON for x in tr); con_f = sum(x[v] == CON for x in fa)
    by = defaultdict(lambda: [0, 0])
    for x in tr:
        by[x["case"]][0] += x[v] == CON; by[x["case"]][1] += 1
    arr = np.array(list(by.values())); boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(arr), len(arr)); M = arr[idx]
        boots.append(100 * M[:, 0].sum() / max(1, M[:, 1].sum()))
    return dict(n=len(L), false_con=100 * con_t / max(1, len(tr)),
                false_con_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                con_precision=100 * con_f / max(1, con_f + con_t), false_sup=100 * sum(x[v] == SUP for x in fa) / max(1, len(fa)),
                detect=100 * con_f / max(1, len(fa)), unv=100 * sum(x[v] == UNV for x in L) / max(1, len(L)))
res = {}
for key in sorted({x["key"] for x in rows}):
    L = [x for x in rows if x["key"] == key]
    res[key] = {v: metrics(L, v) for v in REG}
    res[key]["guaranteed_values"] = 100 * float(np.mean(gshare[key])) if gshare[key] else float("nan")
    Ld = [x for x in L if x["family"] == "derived"]
    res[key]["derived_false_con"] = {v: metrics(Ld, v)["false_con"] for v in REG}
out = dict(fake=a.fake, cases=len(recs), effort=effort, compactness=compact, incompleteness=res)
json.dump(out, open(os.path.join(a.out, "rq5_bpic.json" if not a.fake else "rq5_bpic_FAKE.json"), "w"), indent=1, default=float)

NAMES = dict(assumed="Completeness assumed", guar_acc="Guarantees, accurate", guar_wrong="Guarantees, 50\\% wrong", open="Open world")
PAT = dict(random10="Random 10\\%", random20="Random 20\\%", random30="Random 30\\%", outage="Outage (2--7 days)", late="Late import (1--7 days)")
T = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
     r"Pattern & Verifier & False \textsf{CON} & Prec.\ \textsf{CON} & False \textsf{SUP} & Detection & \textsf{UNV}\\", r"\midrule"]
for k in PAT:
    if k not in res:
        continue
    for i, v in enumerate(REG):
        d = res[k][v]
        T.append(f"{PAT[k] if i == 0 else ''} & {NAMES[v]} & {d['false_con']:.1f} [{d['false_con_ci'][0]:.1f}; {d['false_con_ci'][1]:.1f}] & "
                 f"{d['con_precision']:.1f} & {d['false_sup']:.1f} & {d['detect']:.1f} & {d['unv']:.1f}\\\\")
    T.append(r"\addlinespace")
T[-1] = r"\bottomrule"; T.append(r"\end{tabular}")
if not a.fake:
    open(os.path.join(a.gen, "tab_rq5.tex"), "w").write("\n".join(T) + "\n")
print(json.dumps(dict(effort=effort, compactness=compact), indent=1, default=float))
for k in PAT:
    if k in res:
        print(k, {v: (round(res[k][v]["false_con"], 1), round(res[k][v]["con_precision"], 1), round(res[k][v]["unv"], 1)) for v in REG},
              "guaranteed", round(res[k]["guaranteed_values"], 1))
