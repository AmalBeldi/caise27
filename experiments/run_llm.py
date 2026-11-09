"""E5 — end-to-end evaluation with a language model (to be run where a model is available).

Works with any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp server,
LM Studio, or a hosted API). Example with Ollama:
    ollama pull qwen2.5:7b-instruct
    python3 experiments/run_llm.py --base-url http://localhost:11434/v1 \
        --model qwen2.5:7b-instruct --n-patients 50
Then:
    python3 experiments/score_llm.py results/llm_qwen2.5_7b-instruct.jsonl
Answers are appended to a JSONL file; an interrupted run resumes where it stopped.
`--dry-run` replaces the model by a stub (first lines of the context) to test the
plumbing only: its outputs are marked and must never be reported.
"""
import argparse, json, os, random, sys, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.questions import build_questions
from ehrkg.pipeline import prepare, contexts
from ehrkg import context as C

from ehrkg.prompt import SYSTEM, SYSTEM_V2
from ehrkg.completeness import degrade, Statements

def ask(base, model, key, ctx, q, temperature, max_tokens, timeout=300, system=SYSTEM):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Contexte :\n{ctx}\n\nQuestion : {q}"}]
    r = requests.post(base.rstrip("/") + "/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {key}"} if key else {},
                      json=dict(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens,
                                seed=2027))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--model", default="qwen2.5:7b-instruct")
ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
ap.add_argument("--n-patients", type=int, default=50)
ap.add_argument("--budget", type=int, default=2000, help="budget of the RAG / graph conditions")
ap.add_argument("--budget-text", type=int, default=8000, help="budget of the full-text condition")
ap.add_argument("--conds", default="TEXT-FULL,TEXT-RAG,KG-DG,KG-PPR,OURS,OURS-GSonly,OURS-noHERSE,OURS-noUP")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=200)
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", default=None)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--prompt", default="v1", choices=["v1", "v2"], help="v2 = CAiSE campaign prompt (guarantee markers)")
ap.add_argument("--incomplete", default="none", choices=["none", "gap", "random20", "external"],
                help="degrade each record (complete record = Synthea); gold answers stay those of the complete record")
ap.add_argument("--exclude-from", default=None,
                help="JSONL of an earlier run: its patients are excluded (evaluate on unseen test patients)")
a = ap.parse_args()

REC = load_cohort(a.csv)
pids = sorted(REC)
rng = random.Random(a.seed)
val = set(rng.sample(pids, len(pids) // 2))          # same split as run_symbolic.py
test = [p for p in pids if p not in val]
if a.exclude_from:
    seen = {json.loads(l)["pid"] for l in open(a.exclude_from)}
    order_ = [p for p in random.Random(a.seed + 1).sample(test, len(test)) if p not in seen]
    sel = order_[:a.n_patients]
else:                                                     # original E5 selection (unchanged)
    sel = random.Random(a.seed + 1).sample(test, min(a.n_patients, len(test)))
tag = "DRYRUN" if a.dry_run else a.model.replace("/", "_").replace(":", "_")
out = a.out or f"results/llm_{tag}.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
done = set()
if os.path.exists(out):
    for line in open(out):
        x = json.loads(line); done.add((x["pid"], x["qid"], x["cond"]))
conds = a.conds.split(",")
f = open(out, "a")
n, t0 = 0, time.time()
for pid in sel:
    r = REC[pid]
    if a.incomplete == "none":
        P = prepare(r, "generaliste")
    else:
        pat, rate = ("random", 0.2) if a.incomplete == "random20" else (a.incomplete, 0.0)
        D, removed = degrade(r, pat, random.Random(f"{a.seed}-{pid}-{a.incomplete}"), rate)
        P = prepare(D, "generaliste", Statements.declare(removed, r["patient"]["ref_date"], "accurate"))
    for q in build_questions(r):            # gold answers from the complete record
        cx = contexts(P, q["question"], a.budget, conds=[c for c in conds if c != "TEXT-FULL"])
        if "TEXT-FULL" in conds:
            cx["TEXT-FULL"] = contexts(P, q["question"], a.budget_text, conds=["TEXT-FULL"])["TEXT-FULL"]
        for cond in conds:
            if (pid, q["qid"], cond) in done:
                continue
            ctx, used = cx[cond]
            text = C.to_text(ctx)
            t = time.time()
            if a.dry_run:
                ans = "DRY-RUN " + text[:200]
            else:
                for attempt in range(3):
                    try:
                        ans = ask(a.base_url, a.model, a.api_key, text, q["question"], a.temperature, a.max_tokens,
                                  system=SYSTEM_V2 if a.prompt == "v2" else SYSTEM)
                        break
                    except Exception as e:
                        print("retry", attempt, e, flush=True); time.sleep(5 * (attempt + 1))
                else:
                    ans = None
            row = dict(pid=pid, qid=q["qid"], cond=cond, model=tag, budget=a.budget if cond != "TEXT-FULL" else a.budget_text,
                       ctx_tokens=used, question=q["question"], gold=q["gold"], kind=q["kind"], answer=ans,
                       sufficient=C.sufficient(q["evidence"], C.covered(ctx))[0], latency=time.time() - t,
                       prompt=a.prompt, incomplete=a.incomplete)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); f.flush()
            n += 1
    print(f"patient {sel.index(pid)+1}/{len(sel)}  {n} calls  {time.time()-t0:.0f}s", flush=True)
print("written", out)
