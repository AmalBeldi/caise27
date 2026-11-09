"""Trace pas à pas de l'algorithme 1 (pipeline) pour un patient et une question.

Exemples :
  python3 scripts/demo_algo1.py --patient 0 --question "Quelle est la dernière valeur d'HbA1c et à quelle date ?"
  python3 scripts/demo_algo1.py --patient 38cbc124 --question "Quels sont les traitements antidiabétiques en cours ?" \
      --answer "La patiente est traitée par metformine et sitagliptine."
  # avec un LLM local (Ollama) :
  python3 scripts/demo_algo1.py --patient 0 --question "..." --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct
Chaque étape affiche son entrée, sa sortie et sa durée ; --save écrit G*_RDF en Turtle et le contexte dans out/.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.dg import build_dg
from ehrkg.dgsumm import dgsumm, PROFILES
from ehrkg import rdf as R, context as C
from ehrkg.concepts import link_question, generic_med_label
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--patient", default="0", help="rang dans la cohorte triée, ou fin de l'identifiant")
ap.add_argument("--question", default="Quelle est la dernière valeur d'HbA1c et à quelle date ?")
ap.add_argument("--profile", default="generaliste", choices=list(PROFILES))
ap.add_argument("--budget", type=int, default=500)
ap.add_argument("--answer", default=None, help="réponse à vérifier (sinon : LLM si --base-url, sinon rien)")
ap.add_argument("--base-url", default=None); ap.add_argument("--model", default="qwen2.5:7b-instruct")
ap.add_argument("--save", action="store_true")
a = ap.parse_args()

REC = load_cohort(a.csv); pids = sorted(REC)
pid = pids[int(a.patient)] if a.patient.isdigit() else next(p for p in pids if p.endswith(a.patient))
r = REC[pid]

def step(title):
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))
    return time.perf_counter()

t = step("Entrée"); print(f"patient {pid[-8:]}  date de référence {r['patient']['ref_date']}  profil {a.profile}  budget {a.budget}")
print("question :", a.question)
print("tables :", {k: len(v) for k, v in r.items() if k != "patient"})

t = step("Étape 1 — BuildDG"); G = build_dg(r)
kinds = {}
for _, d in G.nodes(data=True): kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
print(f"|V|={G.number_of_nodes()} |E|={G.number_of_edges()}  {kinds}  ({1000*(time.perf_counter()-t):.1f} ms)")

t = step("Étape 2 — DGsumm (display, filtrate, transformate, calculate, abstract)"); GS = dgsumm(G, a.profile)
print(f"|GS|={GS.number_of_nodes()} nœuds ({1000*(time.perf_counter()-t):.1f} ms)")
for n, d in GS.nodes(data=True):
    if n != "patient":
        print(f"  {d['type']:<17} {n:<32} thème={d['theme']:<16} provenance={len(d['prov'])} faits")

t = step("Étape 3 — RDF-GraphSyn puis HERSE"); tr, roots = R.rdf_graphsyn(GS); ts, iri = R.herse(tr, roots)
print("G_RDF  :", R.stats(tr)); print("G*_RDF :", R.stats(ts), f"({1000*(time.perf_counter()-t):.1f} ms)")

t = step("Étape 4 — Link + BuildContext (+ repli DG)")
L = link_question(a.question); print("concepts liés :", {k: sorted(v) if isinstance(v, set) else v for k, v in L.items() if k != "en"})
items = C.ours_items(GS, ts, iri, tr, roots)
w = PROFILES[a.profile]
for it in sorted(items, key=lambda i: -C.ours_relevance(i, L) * w.get(i["theme"], 1.0))[:12]:
    print(f"  rel={C.ours_relevance(it, L):.1f} w_UP={w.get(it['theme'], 1.0):.1f} jetons={it['ntok']:<4} {it['id']}")
ctx, used = C.ctx_ours(items, a.question, a.budget, w, dg_items=C.dg_items(G))
print(f"contexte retenu : {len(ctx)} éléments, {used} jetons / budget {a.budget}")
text = C.to_text(ctx); print("-" * 72); print(text); print("-" * 72)

answer = a.answer
if answer is None and a.base_url:
    t = step("Étape 5 — QueryLLM")
    import requests
    from ehrkg.prompt import SYSTEM  # same prompt as E5
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Contexte :\n{text}\n\nQuestion : {a.question}"}]
    rr = requests.post(a.base_url.rstrip("/") + "/chat/completions",
                       json=dict(model=a.model, messages=msgs, temperature=0, seed=2027), timeout=300)
    answer = rr.json()["choices"][0]["message"]["content"].strip()
    print(answer, f"({time.perf_counter()-t:.1f} s)")

if answer:
    t = step("Étape 6 — VerifyAndExplain (V4 hiérarchique)")
    extra = sorted({generic_med_label(x) for x in r["medications"].DESCRIPTION})
    X = Extractor(extra_meds=[m for m in extra if len(m) > 3])
    claims = X.extract(answer); res = verify(claims, SummaryKB(ts), RecordKB(G), "V4")
    for c, lab, wit in res:
        print(f"  {lab:<15} {c}")
        for x in wit[:4]: print(f"      témoin : {x}")
    st, score = decide(res); print(f"statut = {st}   score = {score:.2f}")

if a.save:
    os.makedirs("out", exist_ok=True)
    with open(f"out/{pid[-8:]}_gstar.ttl", "w") as f:
        f.write("@prefix ex: <http://example.org/ehr/> .\n@prefix ehr: <http://example.org/ehr-schema#> .\n"
                "@prefix dg: <http://example.org/dg/> .\n@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        for s_, p_, o_ in ts:
            o = o_ if isinstance(o_, str) and o_.split(":")[0] in ("ex", "ehr", "dg", "rdf") else '"%s"' % str(o_).replace('"', "'")
            f.write(f"{s_} {p_} {o} .\n")
    open(f"out/{pid[-8:]}_context.txt", "w").write(text)
    print(f"\nécrit : out/{pid[-8:]}_gstar.ttl, out/{pid[-8:]}_context.txt")
