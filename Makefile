PY ?= python3
MODEL ?= qwen2.5:7b-instruct
BASE_URL ?= http://localhost:11434/v1
LOG ?= data/bpic2017/BPI_Challenge_2017.xes.gz

.PHONY: install test rq1 rq2 rq3 rq4 rq4-fake demo

install:   ## Python dependencies
	$(PY) -m pip install -r requirements.txt

test:      ## unit tests (operators, guarantees, verifier, generic core)
	$(PY) -m unittest discover -s tests -v

demo:      ## trace of the pipeline on one patient
	$(PY) scripts/demo_algo1.py --patient 38cbc124 --question "Quels sont les traitements antidiabétiques en cours ?" --answer "La patiente est traitée par metformine et sitagliptine. Elle présente par ailleurs une anémie." --save

rq1:       ## RQ1 (Table 2, Fig. 2) + empirical check of Proposition 1; no LLM (~6 min)
	$(PY) experiments/run_incompleteness.py --seeds 3
	$(PY) experiments/check_guarantees.py --seeds 3
	$(PY) experiments/analyze_incompleteness.py results/rq4_claims.jsonl paper/generated

rq2:       ## RQ2 (Table 3): scores the released LLM answers; to regenerate them see README
	$(PY) experiments/score_campaign.py results/campaign/*.jsonl --out paper/generated

rq3:       ## RQ3 (Table 4): verifiers V4/V5 on all released LLM answers against the exact ground truth
	$(PY) experiments/verifiers_auto.py results/campaign/*.jsonl --out paper/generated

rq4:       ## RQ4 (Table 5): BPI Challenge 2017 (download the log first, see README)
	$(PY) experiments/run_rq5_bpic.py --log $(LOG) --n-cases 2000 --seeds 3

rq4-fake:  ## RQ4 plumbing on a small synthetic log (not reported)
	$(PY) experiments/run_rq5_bpic.py --fake
