"""External (neural) verifier baselines on LLM answers, given the same record facts as evidence.

Methods
  factscore  FActScore-style (Min et al., 2023), adapted: an LLM decomposes the answer into atomic facts,
             then judges each fact against the evidence: SUPPORTED / CONTRADICTED / NOT ENOUGH INFO
             (three-valued, directly comparable with SUP / CON / UNV).
  minicheck  MiniCheck (Tang et al., 2024): each sentence of the answer is scored against the evidence
             document (supported or not). English-centric model: reported as such.
  judge      LLM-as-a-judge (answer-level OK / KO), same prompt as judge_llm.py, same evidence as the others.

Evidence = facts of the AVAILABLE record linked to the question (same retrieval as "RAG over DG",
3,000 tokens), i.e. the complete record for complete-record answers and the degraded record for
incomplete-record answers. With --with-statements, the evidence also lists, in plain text, the
periods for which the record may be incomplete (the information our verifier gets from completeness
statements): this tests whether a neural verifier can exploit it.

Usage (on a machine with an OpenAI-compatible endpoint, e.g. Ollama on Colab):
  python3 experiments/verifiers_external.py results/campaign/ann_sample.jsonl --method factscore \
      --model qwen2.5:32b-instruct --out results/campaign/ext_factscore.jsonl [--with-statements]
  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  python3 experiments/verifiers_external.py results/campaign/ann_sample.jsonl --method minicheck \
      --out results/campaign/ext_minicheck.jsonl
Resumable: answers already in --out are skipped.
"""
import argparse, json, os, random, re, sys, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.dg import build_dg
from ehrkg import context as C
from ehrkg.completeness import degrade, Statements
from ehrkg.concepts import OBS

ap = argparse.ArgumentParser()
ap.add_argument("answers")
ap.add_argument("--method", required=True, choices=["factscore", "minicheck", "judge"])
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--model", default="qwen2.5:7b-instruct")
ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
ap.add_argument("--budget", type=int, default=3000)
ap.add_argument("--with-statements", action="store_true")
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

REC = load_cohort(a.csv)
rows = [json.loads(l) for l in open(a.answers)]
done = set()
if os.path.exists(a.out):
    done = {(x["pid"], x["qid"], x["cond"], x.get("model"), x.get("incomplete", "none")) for x in map(json.loads, open(a.out))}

LAB = {code: d["label"] for code, d in OBS.items()}
def statements_text(removed):
    by = {}
    for p, y in removed:
        name = ("prescriptions" if p[0] == "MED" else "diagnostics" if p[0] == "COND" else f"mesures {LAB.get(p[1], p[1])}")
        by.setdefault(name, set()).add(y)
    if not by:
        return "Le dossier est complet pour toutes les données."
    return "Données possiblement incomplètes (années concernées) : " + " ; ".join(
        f"{k} ({', '.join(str(y) for y in sorted(v))})" for k, v in sorted(by.items())) + \
        ". Les autres données sont complètes."

EVID = {}
def evidence(pid, question, setting):
    key = (pid, question, setting)
    if key not in EVID:
        r = REC[pid]
        if setting == "none":
            D, removed = r, []
        else:
            pat, rate = ("random", 0.2) if setting == "random20" else (setting, 0.0)
            D, removed = degrade(r, pat, random.Random(f"{a.seed}-{pid}-{setting}"), rate)
        G = build_dg(D)
        ctx, _ = C.ctx_kg_dg(C.dg_items(G), question, a.budget)
        txt = f"Date de référence du dossier : {r['patient']['ref_date']}.\n" + C.to_text(ctx)
        if a.with_statements:
            txt += "\n" + statements_text(removed)
        EVID[key] = txt
    return EVID[key]

def chat(prompt, max_tokens=600):
    if a.dry_run:
        return '["fait"]' if "JSON" in prompt else "NOT ENOUGH INFO"
    for attempt in range(3):
        try:
            r = requests.post(a.base_url.rstrip("/") + "/chat/completions", timeout=300,
                              headers={"Authorization": f"Bearer {a.api_key}"} if a.api_key else {},
                              json=dict(model=a.model, temperature=0, seed=2027, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}]))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("retry", attempt, e, flush=True); time.sleep(5 * (attempt + 1))
    return None

DECOMP = ("Décomposez la réponse suivante en faits atomiques indépendants (une seule information vérifiable par fait, "
          "avec sa valeur et sa date s'il y en a). Répondez uniquement par une liste JSON de chaînes.\n\n"
          "Question : {q}\nRéponse : {a}")
CHECK = ("Vous vérifiez un fait à partir d'extraits du dossier d'un patient.\n\nExtraits du dossier :\n{e}\n\n"
         "Fait : {f}\n\nLe fait est-il SUPPORTED (établi par les extraits), CONTRADICTED (contredit par les extraits) "
         "ou NOT ENOUGH INFO (les extraits ne permettent pas de conclure) ? Répondez par un seul de ces trois libellés.")

JUDGE = ("Vous vérifiez la réponse d'un assistant à une question sur le dossier d'un patient.\n"
         "Faits extraits du dossier :\n{e}\n\nQuestion : {q}\nRéponse : {a}\n\n"
         "La réponse contient-elle au moins une affirmation fausse ou non étayée par ces faits ? "
         "Répondez uniquement par OK (toutes les affirmations sont étayées) ou KO.")

def parse_list(t):
    if not t:
        return []
    m = re.search(r"\[.*\]", t, re.S)
    try:
        x = json.loads(m.group(0)) if m else []
        return [str(s) for s in x if str(s).strip()]
    except Exception:
        return [s.strip("-• ").strip() for s in t.splitlines() if s.strip()]

def label(t):
    t = (t or "").upper()
    return "CON" if "CONTRADICT" in t else "SUP" if "SUPPORTED" in t else "UNV"

scorer = None
if a.method == "minicheck" and not a.dry_run:
    from minicheck.minicheck import MiniCheck                 # pip install from the MiniCheck repository
    scorer = MiniCheck(model_name="flan-t5-large", cache_dir="./ckpts")

f = open(a.out, "a"); n = 0; t0 = time.time()
for r in rows:
    k = (r["pid"], r["qid"], r["cond"], r.get("model"), r.get("incomplete", "none"))
    if k in done or not r.get("answer"):
        continue
    ev = evidence(r["pid"], r["question"], r.get("incomplete", "none"))
    if a.method == "factscore":
        facts = parse_list(chat(DECOMP.format(q=r["question"], a=r["answer"]))) or [r["answer"]]
        labs = [label(chat(CHECK.format(e=ev, f=fa), max_tokens=10)) for fa in facts]
    elif a.method == "judge":
        raw = chat(JUDGE.format(e=ev, q=r["question"], a=r["answer"]), max_tokens=5) or ""
        facts, labs = [r["answer"]], ["CON" if re.search(r"\bKO\b", raw.upper()) else "SUP" if "OK" in raw.upper() else "UNV"]
    else:
        facts = [s for s in re.split(r"(?<=[.;!?])\s+", r["answer"]) if s.strip()] or [r["answer"]]
        if a.dry_run:
            labs = ["UNV"] * len(facts)
        else:
            pred, prob, _, _ = scorer.score(docs=[ev] * len(facts), claims=facts)
            labs = ["SUP" if p == 1 else "UNV" for p in pred]       # MiniCheck is binary: supported or not
    verdict = "KO" if "CON" in labs else ("OK" if all(l == "SUP" for l in labs) else "UNVERIFIED")
    out = dict(pid=r["pid"], qid=r["qid"], cond=r["cond"], model=r.get("model"), incomplete=r.get("incomplete", "none"),
               method=a.method + ("+statements" if a.with_statements else ""), verifier_model="flan-t5-large" if a.method == "minicheck" else a.model,
               facts=facts, labels=labs, verdict=verdict)
    f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush(); n += 1
    if n % 25 == 0:
        print(f"{n} answers  {time.time()-t0:.0f}s", flush=True)
print("written", a.out, n)
