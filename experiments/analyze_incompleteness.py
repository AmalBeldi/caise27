"""RQ4 tables and figure from results/rq4_claims.jsonl.

Claim-level metrics, 95% CIs by patient-level bootstrap (2,000 resamples;
claims of a patient are resampled together across seeds).
Writes paper/generated/tab_rq4.tex, paper/generated/fig_rq4.pdf and
results/rq4_table.json.

  python3 experiments/analyze_incompleteness.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/rq4_claims.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/generated"
os.makedirs(OUT, exist_ok=True)
CON, UNV, SUP, SUPO = "contredit", "non_verifiable", "supporte", "supporte_non_garanti"
VER = ["V2H", "V4", "V5a", "V5w"]
NAMES = {"V2H": "Open world", "V4": "Completeness assumed (V4)",
         "V5a": "Guarantees, accurate (V5)", "V5w": "Guarantees, 50\\% wrong"}
PAT = [("random10", "Random 10\\%"), ("random20", "Random 20\\%"), ("random30", "Random 30\\%"),
       ("gap", "Care gap"), ("external", "External drug")]

rows = [json.loads(l) for l in open(SRC)]
def key(x):
    return x["pattern"] + (str(int(round(100 * x["rate"]))) if x["pattern"] == "random" else "")

# per (pattern, patient) counters -> bootstrap over patients
C = defaultdict(lambda: defaultdict(lambda: np.zeros(8)))
# counters: 0 n_true, 1 n_false, 2 n_all, 3 con|true, 4 con|false, 5 unv, 6 sup_strict|false, 7 decided-correct
for x in rows:
    for v in VER:
        a = C[(key(x), v)][x["pid"]]
        l = x[v]
        a[2] += 1
        if x["truth"]:
            a[0] += 1; a[3] += l == CON; a[7] += l in (SUP, SUPO)
        else:
            a[1] += 1; a[4] += l == CON; a[6] += l == SUP; a[7] += l == CON
        a[5] += l == UNV

rng = np.random.default_rng(2027)
def stat(M, num, den):
    return 100 * M[:, num].sum() / max(1, M[:, den].sum())
def ci(arr, num, den, B=2000):
    n = len(arr); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n); M = arr[idx]
        vals.append(stat(M, num, den))
    return np.percentile(vals, [2.5, 97.5])

table = {}
for k, _ in PAT:
    for v in VER:
        arr = np.array(list(C[(k, v)].values()))
        if not len(arr):
            continue
        d = {}
        for name, num, den in [("false_con", 3, 0), ("detect", 4, 1), ("unv", 5, 2), ("false_sup", 6, 1),
                               ("decided_correct", 7, 2)]:
            d[name] = stat(arr, num, den); d[name + "_ci"] = list(ci(arr, num, den, B=1000 if name != "false_con" else 2000))
        d["con_precision"] = 100 * arr[:, 4].sum() / max(1, arr[:, 3].sum() + arr[:, 4].sum())
        d["n_claims"] = int(arr[:, 2].sum()); d["patients"] = len(arr)
        table[f"{k}|{v}"] = d
json.dump(table, open(os.path.join(os.path.dirname(SRC), "rq4_table.json"), "w"), indent=1)

# ---------------------------------------------------------------- LaTeX table
L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
     r"Incompleteness & Verifier & False \textsf{CON} & Prec.\ of & False \textsf{SUP} & Detection & \textsf{UNV}\\",
     r" & & (true claims) & \textsf{CON} & (false claims) & (false claims) & (all)\\", r"\midrule"]
for k, lab in PAT:
    first = True
    for v in ["V4", "V5a", "V5w", "V2H"]:
        d = table.get(f"{k}|{v}")
        if not d:
            continue
        fc = f"{d['false_con']:.1f} [{d['false_con_ci'][0]:.1f}; {d['false_con_ci'][1]:.1f}]"
        L.append(f"{lab if first else ''} & {NAMES[v]} & {fc} & {d['con_precision']:.1f} & {d['false_sup']:.1f} & {d['detect']:.1f} & {d['unv']:.1f}\\\\")
        first = False
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(OUT, "tab_rq4.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure (two panels, one axis each)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "pdf.fonttype": 42})
STYLE = {"V4": ("#2a78d6", "o", "-"), "V5a": ("#eb6834", "s", "-"), "V5w": ("#1baf7a", "^", "--"),
         "V2H": ("#eda100", "D", ":")}
SHORT = {"V4": "Completeness assumed", "V5a": "Guarantees (accurate)", "V5w": "Guarantees (50% wrong)",
         "V2H": "Open world"}
xs = [10, 20, 30]
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
for ax, (metric, ylab) in zip(axes, [("false_con", "True claims contradicted (%)"), ("unv", "Claims left unverifiable (%)")]):
    for v in ["V4", "V2H", "V5w", "V5a"]:
        col, mk, ls = STYLE[v]
        ys = [table[f"random{x}|{v}"][metric] for x in xs]
        lo = [table[f"random{x}|{v}"][metric + "_ci"][0] for x in xs]
        hi = [table[f"random{x}|{v}"][metric + "_ci"][1] for x in xs]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=col, marker=mk, linestyle=ls, linewidth=1.5, markersize=4, label=SHORT[v])
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs]); ax.set_xlabel("Random loss of facts")
    ax.set_ylabel(ylab); ax.grid(axis="y", color="#e6e5e0", linewidth=0.6); ax.set_ylim(bottom=0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=2, handlelength=2.2, columnspacing=1.0)
fig.tight_layout(w_pad=1.5, rect=(0, 0.17, 1, 1))
fig.savefig(os.path.join(OUT, "fig_rq4.pdf")); fig.savefig(os.path.join(OUT, "fig_rq4.png"), dpi=200)
print("written", OUT)
