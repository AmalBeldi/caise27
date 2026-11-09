"""Score the CAiSE LLM campaign (several models, serialisations S1/S2/S3, complete and incomplete records).

Usage:
  python3 experiments/score_campaign.py results/campaign/*.jsonl [--ref OURS-S3] [--out paper/generated]

Per (model, record setting, condition):
  acc        answer correct w.r.t. the COMPLETE record (deterministic rules, ehrkg/scoring.py), 95% CI (patient bootstrap)
  acc_fact / acc_agg / acc_out   by question group
  contra     share of extracted claims contradicted by the complete record (hierarchical verifier on D*)
  abst       "Information non disponible"
  hedged     answer says "d'après les données disponibles"
  conf_wrong wrong, not abstained and not hedged ("confidently wrong")
  shown_ok   the verifier the system would run (V5 on the available record, accurate statements) shows the answer as OK
  risk_ok    share of wrong answers among those shown as OK (residual risk)
  tokens     median context tokens
Paired comparison with the reference condition: exact McNemar test, Holm correction per (model, setting).
If statsmodels is installed, a GEE logistic model (exchangeable, clustered by patient) is also fitted per setting.
"""
import argparse, json, os, random, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
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
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--ref", default="OURS-S3")
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", default="paper/generated")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = [json.loads(l) for f in a.files for l in open(f)]
rows = [r for r in rows if r.get("answer") is not None and r.get("model") != "DRYRUN"]
REC = load_cohort(a.csv)
extra = sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION})
X = Extractor(extra_meds=[m for m in extra if len(m) > 3])

def kbs(rec):
    G = build_dg(rec); GS = dgsumm(G, "generaliste")
    tr, roots = R.rdf_graphsyn(GS); ts, _ = R.herse(tr, roots)
    return SummaryKB(ts), RecordKB(G)

CACHE = {}
def kb_for(pid, setting):
    k = (pid, setting)
    if k not in CACHE:
        r = REC[pid]
        if setting == "none":
            KS, KR = kbs(r); S = Statements(set(), r["patient"]["ref_date"])
        else:
            pat, rate = ("random", 0.2) if setting == "random20" else (setting, 0.0)
            D, removed = degrade(r, pat, random.Random(f"{a.seed}-{pid}-{setting}"), rate)   # same draw as run_llm.py
            KS, KR = kbs(D); S = Statements.declare(removed, r["patient"]["ref_date"], "accurate")
        CACHE[k] = (KS, KR, S)
    return CACHE[k]

per = defaultdict(list)        # (model, setting, cond) -> list of dicts
for r in rows:
    setting = r.get("incomplete", "none")
    ok = correct(r, X)
    ans = r["answer"]; al = ans.lower()
    claims = X.extract(ans)
    KSs, KRs, _ = kb_for(r["pid"], "none")
    truth = [l for _, l, _ in verify(claims, KSs, KRs, "V4")]            # against the complete record
    KS, KR, S = kb_for(r["pid"], setting)
    v4 = [l for _, l, _ in verify(claims, KS, KR, "V4")]
    v5 = [guard(c, l, KR, S) for c, l in zip(claims, v4)]                  # what the system would show
    shown_ok = bool(v5) and all(l in (SUP, SUPO) for l in v5)
    abst = "non disponible" in al
    hedged = "données disponibles" in al or "donnees disponibles" in al
    grp = "out" if r["qid"].startswith("QX") else ("agg" if r["qid"] in ("Q3", "Q10") else "fact")
    per[(r["model"], setting, r["cond"])].append(dict(
        pid=r["pid"], qid=r["qid"], ok=ok, grp=grp, abst=abst, hedged=hedged,
        conf_wrong=(not ok) and (not abst) and (not hedged), n_claims=len(truth),
        n_con=sum(l == CON for l in truth), shown_ok=shown_ok, tok=r["ctx_tokens"]))

rng = np.random.default_rng(a.seed)
def boot_ci(items, B=2000):
    by = defaultdict(list)
    for x in items:
        by[x["pid"]].append(x["ok"])
    P = list(by.values()); n = len(P)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        s = [v for i in idx for v in P[i]]
        vals.append(100 * np.mean(s))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def pct(xs):
    return 100 * float(np.mean(xs)) if xs else float("nan")

summary = {}
for (m, st, c), L in sorted(per.items()):
    okshown = [x for x in L if x["shown_ok"]]
    d = dict(n=len(L), acc=pct([x["ok"] for x in L]), acc_ci=boot_ci(L),
             acc_fact=pct([x["ok"] for x in L if x["grp"] == "fact"]),
             acc_agg=pct([x["ok"] for x in L if x["grp"] == "agg"]),
             acc_out=pct([x["ok"] for x in L if x["grp"] == "out"]),
             contra=100 * sum(x["n_con"] for x in L) / max(1, sum(x["n_claims"] for x in L)),
             abst=pct([x["abst"] for x in L]), hedged=pct([x["hedged"] for x in L]),
             conf_wrong=pct([x["conf_wrong"] for x in L]),
             shown_ok=pct([x["shown_ok"] for x in L]),
             risk_ok=pct([not x["ok"] for x in okshown]) if okshown else float("nan"),
             tokens=float(np.median([x["tok"] for x in L])))
    summary[f"{m}|{st}|{c}"] = d

# paired McNemar vs reference condition, Holm per (model, setting)
for (m, st) in sorted({(k[0], k[1]) for k in per}):
    ref = {(x["pid"], x["qid"]): x["ok"] for x in per.get((m, st, a.ref), [])}
    if not ref:
        continue
    tests = []
    for (mm, ss, c), L in per.items():
        if (mm, ss) != (m, st) or c == a.ref:
            continue
        other = {(x["pid"], x["qid"]): x["ok"] for x in L}
        keys = ref.keys() & other.keys()
        b = sum(ref[k] and not other[k] for k in keys); cc = sum(other[k] and not ref[k] for k in keys)
        p = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
        tests.append((c, b, cc, p))
    tests.sort(key=lambda t: t[3])
    k = len(tests); running = 0.0
    for i, (c, b, cc, p) in enumerate(tests):
        running = max(running, min(1.0, (k - i) * p))
        summary[f"{m}|{st}|{c}"].update(mcnemar_b=b, mcnemar_c=cc, p=p, p_holm=running, ref=a.ref)

# optional GEE (needs statsmodels; available on Colab)
try:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    for (m, st) in sorted({(k[0], k[1]) for k in per}):
        df = pd.DataFrame([dict(x, cond=c) for (mm, ss, c), L in per.items() if (mm, ss) == (m, st) for x in L])
        if df.cond.nunique() < 2 or a.ref not in set(df.cond):
            continue
        df["y"] = df.ok.astype(int)
        res = smf.gee(f"y ~ C(cond, Treatment('{a.ref}')) + C(qid)", "pid", df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        summary[f"{m}|{st}|_gee"] = {k: dict(coef=float(v), p=float(res.pvalues[k]),
                                             odds_ratio=float(np.exp(v))) for k, v in res.params.items() if "cond" in k}
except Exception as e:                       # statsmodels missing or model not estimable
    summary["_gee_note"] = f"GEE skipped: {type(e).__name__}: {e}"

json.dump(summary, open(os.path.join(a.out, "summary_campaign.json"), "w"), indent=1, default=float)

# per-answer rows and paired differences (patient-level bootstrap) between key conditions
with open(os.path.join(a.out, "campaign_answers_scored.jsonl"), "w") as f:
    for (m, st, c), L in per.items():
        for x in L:
            f.write(json.dumps(dict(x, model=m, setting=st, cond=c), default=bool) + "\n")
PAIRS = [("OURS-S3", "OURS-S2"), ("OURS-S2", "OURS"), ("OURS-S3", "KG-DG")]
diffs = {}
for (m, st) in sorted({(k[0], k[1]) for k in per}):
    for c1, c2 in PAIRS:
        A1 = {(x["pid"], x["qid"]): x for x in per.get((m, st, c1), [])}
        A2 = {(x["pid"], x["qid"]): x for x in per.get((m, st, c2), [])}
        keys = sorted(A1.keys() & A2.keys())
        if not keys:
            continue
        pids = sorted({k[0] for k in keys}); byp = defaultdict(list)
        for k in keys:
            byp[k[0]].append(k)
        def stats(sel):
            a1 = [A1[k] for k in sel]; a2 = [A2[k] for k in sel]
            acc = 100 * (np.mean([x["ok"] for x in a1]) - np.mean([x["ok"] for x in a2]))
            cw = 100 * (np.mean([x["conf_wrong"] for x in a1]) - np.mean([x["conf_wrong"] for x in a2]))
            ct = 100 * (sum(x["n_con"] for x in a1) / max(1, sum(x["n_claims"] for x in a1))
                        - sum(x["n_con"] for x in a2) / max(1, sum(x["n_claims"] for x in a2)))
            return acc, cw, ct
        point = stats(keys); boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(pids), len(pids))
            boots.append(stats([k for i in idx for k in byp[pids[i]]]))
        boots = np.array(boots)
        diffs[f"{m}|{st}|{c1}-{c2}"] = {name: dict(diff=float(point[j]), ci=[float(np.percentile(boots[:, j], 2.5)),
                                                                               float(np.percentile(boots[:, j], 97.5))])
                                        for j, name in enumerate(("acc", "conf_wrong", "contra"))}
json.dump(diffs, open(os.path.join(a.out, "campaign_paired_diffs.json"), "w"), indent=1)
for k, v in diffs.items():
    print("DIFF", k, {n: f"{d['diff']:+.1f} [{d['ci'][0]:+.1f}; {d['ci'][1]:+.1f}]" for n, d in v.items()})

# LaTeX tables ---------------------------------------------------------------
NAMES = {"TEXT-FULL": "Full text", "TEXT-RAG": "Text RAG (BM25)", "KG-DG": "RAG over $DG$", "KG-PPR": "PPR over $DG$",
         "OURS": "Summary, S1 grouped", "OURS-S2": "Summary, S2 explicit", "OURS-S3": "Summary, S3 explicit + guarantees",
         "OURS-noHERSE": "Summary, blank nodes", "OURS-GSonly": "Summary, no fallback", "OURS-noUP": "Summary, uniform profile"}
def star(d):
    p = d.get("p_holm")
    return "" if p is None else ("$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "")
for st, fname in [("none", "tab_campaign_complete.tex"), ("gap", "tab_campaign_gap.tex"), ("random20", "tab_campaign_random20.tex")]:
    keys = [k for k in summary if k.count("|") == 2 and k.split("|")[1] == st and not k.endswith("_gee")]
    if not keys:
        continue
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Model & Condition & Acc.\ [95\% CI] & Fact. & Aggr. & Out & Contra. & Conf.\ wrong\\", r"\midrule"]
    for m in sorted({k.split("|")[0] for k in keys}):
        first = True
        for c in NAMES:
            d = summary.get(f"{m}|{st}|{c}")
            if not d:
                continue
            L.append(f"{m if first else ''} & {NAMES[c]} & {d['acc']:.1f}{star(d)} [{d['acc_ci'][0]:.1f}; {d['acc_ci'][1]:.1f}] & "
                     f"{d['acc_fact']:.1f} & {d['acc_agg']:.1f} & {d['acc_out']:.1f} & {d['contra']:.1f} & {d['conf_wrong']:.1f}\\\\")
            first = False
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
    open(os.path.join(a.out, fname), "w").write("\n".join(L) + "\n")

for k, d in summary.items():
    if k.count("|") == 2 and not k.endswith("_gee"):
        print(f"{k:45s} n={d['n']:4d} acc={d['acc']:5.1f} contra={d['contra']:5.1f} conf_wrong={d['conf_wrong']:5.1f} "
              f"hedged={d['hedged']:5.1f} risk_ok={d['risk_ok']:5.1f} tok={d['tokens']:.0f}"
              + (f" p_holm={d['p_holm']:.3g}" if "p_holm" in d else ""))
print("written", a.out)
