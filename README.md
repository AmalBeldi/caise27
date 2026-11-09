# Summaries with Guarantees — artifact (anonymous)

Code, data and LLM outputs for the paper *Summaries with Guarantees: Completeness- and Provenance-Preserving
Graph Summarisation for Verifiable LLM-Based Information Systems*.

## Quick start
```bash
pip install -r requirements.txt
make test     # 27 unit tests (< 1 min)
make rq1      # RQ1: verification under controlled incompleteness + check of Proposition 1, no LLM (~6 min)
make rq2      # RQ2: scores the 17,578 released LLM answers (results/campaign/)
make rq3      # RQ3: verifiers V4 / V5 on the same answers against the exact ground truth
make rq4 LOG=<path to BPI_Challenge_2017.xes.gz>   # RQ4: second instantiation (process event log)
```
Outputs (JSON summaries used for the tables) are written to `paper/generated/`.

## Regenerating the LLM answers (RQ2)
`experiments/run_llm.py` queries any OpenAI-compatible endpoint (we used Ollama with
`qwen2.5:7b-instruct` and `llama3.1:8b`, `num_ctx` 16384, temperature 0, one GPU), e.g.
```bash
python3 experiments/run_llm.py --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct \
    --prompt v2 --incomplete gap --n-patients 50 \
    --conds KG-DG,OURS,OURS-S2,OURS-S3 --out results/campaign/qwen7_gap.jsonl
```
Options: `--incomplete none|gap|random20|external`, conditions include the baselines and the serialisations
S1/S2/S3 (`OURS`, `OURS-S2`, `OURS-S3`). The released answers are in `results/campaign/` with their run logs.

Neural baselines (RQ3, Table 7): `experiments/verifiers_external.py` runs the LLM judge and the FActScore-style
verifier on the 240-answer sample `results/rq3_sample/sample_240.jsonl` (answers of each LLM judged by the other);
outputs are in `results/external/`. Table 7 is recomputed by
`python3 experiments/verifiers_auto.py results/campaign/*.jsonl --sample results/rq3_sample/sample_240.jsonl --external results/external/*.jsonl`.

## Layout
| Path | Content |
|---|---|
| `ehrkg/` | EHR instantiation: data graph (`dg.py`), summary operators (`dgsumm.py`), RDF publication without blank nodes (`rdf.py`), contexts and serialisations S1/S2/S3 (`context.py`), claim extraction and hierarchical verification (`verify.py`), completeness statements, guarantee propagation and verifier V5 (`completeness.py`), scoring rules (`scoring.py`) |
| `gsum/` | domain-independent core (`core.py`) and the BPI Challenge 2017 configuration (`bpic.py`) |
| `experiments/` | `run_incompleteness.py`, `check_guarantees.py`, `analyze_incompleteness.py` (RQ1); `run_llm.py`, `score_campaign.py` (RQ2); `verifiers_auto.py` (RQ3); `run_rq5_bpic.py` (RQ4) |
| `data/synthea_t2dm_cohort/` | the 172-patient Synthea cohort (seeds `-s 2027 -cs 2027`) |
| `results/` | LLM answers and logs (`campaign/`), RQ1 claim-level outputs, Proposition 1 check, RQ4 outputs |

## Data
All EHRs are synthetic (Synthea); no real patient data is used. BPI Challenge 2017 is public
(4TU.ResearchData, doi:10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) and is not redistributed;
place it at `data/bpic2017/BPI_Challenge_2017.xes.gz`.

Note on file names: `results/rq4_*` and `prop1_check.json` hold the outputs of RQ1 in the paper, and
`results/rq5_bpic.json` those of RQ4 (internal numbering of earlier drafts).
