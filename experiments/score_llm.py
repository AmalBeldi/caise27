"""Score the answers produced by run_llm.py and write paper/generated/tab_llm.tex.

Metrics per condition:
  Exact.   answer correct w.r.t. the gold computed from the raw tables (deterministic rules below)
  Abst.    "Information non disponible"
  Halluc.  share of extracted claims that are contradicted or unverifiable against the record (V4)
  Unsafe   answers with at least one contradicted drug/condition claim (invented treatment or disease,
           in the spirit of the binary safety metric of Thio et al., 2026)
  OK->err  with the verifier: answers labelled OK that are wrong (residual risk once VerifyAndExplain filters)
  Couv.    answers whose claims are all supported, i.e. fully covered by a proof sub-graph G_exp
Usage: python3 experiments/score_llm.py results/llm_<model>.jsonl [more files]
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.pipeline import prepare
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide, parse_dates, SUP, CON, UNV, COND_FR, DATE_RE
from ehrkg.concepts import generic_med_label, COMPLICATIONS, OBS

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

files = sys.argv[1:]
rows = [json.loads(l) for f in files for l in open(f)]
if any(r["model"] == "DRYRUN" for r in rows):
    print("WARNING: dry-run rows present — plumbing test only, not reportable")
REC = load_cohort(os.environ.get("EHR_CSV", "data/synthea_t2dm_cohort"))
extra = sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION})
X = Extractor(extra_meds=[m for m in extra if len(m) > 3])
KB = {}
agg = defaultdict(list)
paired = defaultdict(dict)
for r in rows:
    if r["answer"] is None:
        continue
    if r["pid"] not in KB:
        P = prepare(REC[r["pid"]]); KB[r["pid"]] = (SummaryKB(P["rdf_star"]), RecordKB(P["G"]))
    KS, KR = KB[r["pid"]]
    ok = correct(r, X)
    claims = X.extract(r["answer"])
    res = verify(claims, KS, KR, "V4")
    labs = [l for _, l, _ in res]
    status, _ = decide(res)
    unsafe = any(l == CON and c[0] in ("DRUG", "COND") for c, l, _ in res)
    key = (r["model"], r["cond"])
    agg[key + ("exact",)].append(ok)
    grp = "hors" if r["qid"].startswith("QX") else ("agg" if r["qid"] in ("Q3", "Q10") else "fact")
    agg[key + ("exact", grp)].append(ok)
    if labs:   # automatic agreement of V4 with the deterministic scorer, on every answer with claims
        agg[(r["model"], "ALL", "vauto")].append((status != "OK", not ok))
    paired[key][(r["pid"], r["qid"])] = ok
    agg[key + ("abst",)].append("non disponible" in r["answer"].lower())
    agg[key + ("claims",)] += labs
    agg[key + ("unsafe",)].append(unsafe)
    agg[key + ("ok_wrong",)].append(status == "OK" and not ok)
    agg[key + ("covered",)].append(bool(labs) and all(l == SUP for l in labs))
    agg[key + ("tok",)].append(r["ctx_tokens"])
    agg[key + ("lat",)].append(r["latency"])
NAMES = {"TEXT-FULL": "Texte intégral", "TEXT-RAG": "RAG textuel (BM25)", "KG-DG": "RAG sur DG", "KG-PPR": "PPR sur DG (HippoRAG)",
         "OURS": "Notre approche", "OURS-GSonly": "-- sans repli DG", "OURS-noHERSE": "-- sans HERSE",
         "OURS-noUP": "-- sans UP"}
models = sorted({k[0] for k in agg})
L = ["\\begin{tabular}{llrrrrrrr}", "\\hline",
     "Modèle & Condition & Exact. & Abst. & Halluc. & Unsafe & OK$\\rightarrow$err & Couv. & Jetons \\\\", "\\hline"]
summary = {}
for m in models:
    for c in NAMES:
        k = (m, c)
        if not agg[k + ("exact",)]:
            continue
        cl = agg[k + ("claims",)]
        hall = 100 * np.mean([l != SUP for l in cl]) if cl else 0
        v = dict(exact=100 * np.mean(agg[k + ("exact",)]), abst=100 * np.mean(agg[k + ("abst",)]), halluc=hall,
                 unsafe=100 * np.mean(agg[k + ("unsafe",)]), ok_wrong=100 * np.mean(agg[k + ("ok_wrong",)]),
                 covered=100 * np.mean(agg[k + ("covered",)]), tokens=float(np.median(agg[k + ("tok",)])),
                 n=len(agg[k + ("exact",)]),
                 **{f"exact_{g}": 100 * np.mean(agg[k + ("exact", g)]) for g in ("fact", "agg", "hors")
                    if agg[k + ("exact", g)]})
        summary[f"{m}|{c}"] = v
        L.append(f"{m} & {NAMES[c]} & {v['exact']:.1f} & {v['abst']:.1f} & {v['halluc']:.1f} & {v['unsafe']:.1f} & "
                 f"{v['ok_wrong']:.1f} & {v['covered']:.1f} & {v['tokens']:.0f} \\\\")
L += ["\\hline", "\\end{tabular}"]
print("\n".join(L))
for m in models:
    pr = agg[(m, "ALL", "vauto")]
    if pr:
        tp = sum(1 for a_, b_ in pr if a_ and b_); fp = sum(1 for a_, b_ in pr if a_ and not b_)
        fn = sum(1 for a_, b_ in pr if not a_ and b_)
        P_ = tp / (tp + fp) if tp + fp else 0; R_ = tp / (tp + fn) if tp + fn else 0
        summary[f"{m}|verif_auto"] = dict(n=len(pr), precision=100 * P_, recall=100 * R_,
                                          f1=100 * (2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0))
        print(f"{m} : V4 vs score déterministe sur {len(pr)} réponses : P={100*P_:.1f} R={100*R_:.1f}")

# ---- statistics: patient-level bootstrap CI of exactness, exact McNemar test OURS vs each condition
from math import comb
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def boot_ci(d, B=2000, seed=0):
    rng = np.random.default_rng(seed); pids = sorted({p for p, _ in d})
    by = defaultdict(list)
    for (p, q), v in d.items(): by[p].append(v)
    vals = []
    for _ in range(B):
        smp = rng.choice(pids, len(pids)); x = [v for p in smp for v in by[p]]
        vals.append(100 * np.mean(x))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
LS = ["\\begin{tabular}{llrrr}", "\\hline", "Modèle & Condition & Exact. [IC 95\\,\\%] & $b/c$ & $p$ (McNemar) \\\\", "\\hline"]
for m in models:
    ref_ = paired[(m, "OURS")]
    for c in NAMES:
        d = paired.get((m, c))
        if not d: continue
        lo, hi = boot_ci(d)
        if c == "OURS":
            LS.append(f"{m} & {NAMES[c]} & {100*np.mean(list(d.values())):.1f} [{lo:.1f} ; {hi:.1f}] & -- & -- \\\\"); continue
        common = set(d) & set(ref_)
        b = sum(1 for k in common if ref_[k] and not d[k]); cc = sum(1 for k in common if d[k] and not ref_[k])
        pv = mcnemar(b, cc); summary[f"{m}|{c}"].update(ci=[lo, hi], mcnemar_b=b, mcnemar_c=cc, p=pv)
        LS.append(f"{m} & {NAMES[c]} & {100*np.mean(list(d.values())):.1f} [{lo:.1f} ; {hi:.1f}] & {b}/{cc} & {pv:.2g} \\\\")
LS += ["\\hline", "\\end{tabular}"]
print("\n".join(LS))
if "DRYRUN" in models:
    sys.exit("dry-run rows: table not written")
os.makedirs("paper/generated", exist_ok=True)
open("paper/generated/tab_llm.tex", "w").write("\n".join(L))
open("paper/generated/tab_llm_stats.tex", "w").write("\n".join(LS))
json.dump(summary, open("paper/generated/summary_llm.json", "w"), indent=1)
