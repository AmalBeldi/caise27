"""Test of the E5 answer scorer: reference answers are judged correct, injected errors are judged wrong."""
import os, random, sys, unittest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from ehrkg.data import load_cohort
from ehrkg.questions import build_questions
from ehrkg.perturb import perturbations
from ehrkg.verify import Extractor

src = open(os.path.join(ROOT, "experiments", "score_llm.py"), encoding="utf-8").read().split("files = sys.argv[1:]")[0]
NS = {"__file__": os.path.join(ROOT, "experiments", "score_llm.py")}
exec(src, NS)
correct = NS["correct"]
N = int(os.environ.get("EHRKG_TEST_N", "20"))

class TestScorer(unittest.TestCase):
    def test_references_and_errors(self):
        REC = load_cohort(); X = Extractor(); rng = random.Random(0)
        wrong_ok = total_err = 0
        for p in sorted(REC)[:N]:
            for q in build_questions(REC[p]):
                if q["qid"].startswith("QX"):
                    continue
                for pt, a in perturbations(q, REC[p], rng):
                    ok = correct(dict(answer=a, gold=q["gold"], kind=q["kind"]), X)
                    if pt == "P0":
                        self.assertTrue(ok, (q["qid"], a))
                    elif pt != "P6":
                        total_err += 1; wrong_ok += ok
        self.assertLessEqual(wrong_ok / total_err, 0.01)

if __name__ == "__main__":
    unittest.main(verbosity=2)
