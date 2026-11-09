"""Tests of each step of Algorithm 1 and of Algorithm 2 (VerifyAndExplain).

Run:  python3 -m unittest -v tests/test_algorithms.py   (or: pytest tests/)
Uses the bundled cohort (data/synthea_t2dm_cohort); EHRKG_TEST_N patients (default 20).
Each test states the property of the paper that it checks.
"""
import os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.questions import build_questions
from ehrkg.pipeline import prepare, contexts, CONDITIONS
from ehrkg.concepts import OBS, condition_theme, link_question
from ehrkg import context as C, rdf as R
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide, SUP, CON, UNV
from ehrkg.perturb import perturbations

N = int(os.environ.get("EHRKG_TEST_N", "20"))
REC = load_cohort()
PIDS = sorted(REC)[:N]
PREP = {p: prepare(REC[p]) for p in PIDS}
QS = {p: build_questions(REC[p]) for p in PIDS}


class T1_BuildDG(unittest.TestCase):
    """Etape 1 : projection du DME dans DG, provenance et ordre temporel."""
    def test_every_fact_is_a_node_with_concept(self):
        for p in PIDS:
            G, r = PREP[p]["G"], REC[p]
            n_rows = sum(len(r[t]) for t in ("observations", "conditions", "medications", "procedures"))
            facts = [n for n, d in G.nodes(data=True) if d["kind"] in ("Observation", "Condition", "Medication", "Procedure")]
            self.assertEqual(len(facts), n_rows)
            for n in facts[:200]:
                self.assertTrue(any(k == "instanceOf" for _, _, k in G.out_edges(n, data="label")))

    def test_encounters_are_chained(self):
        for p in PIDS:
            G = PREP[p]["G"]
            n_enc = sum(1 for _, d in G.nodes(data=True) if d["kind"] == "Encounter")
            n_next = sum(1 for _, _, k in G.edges(data="label") if k == "NEXT")
            self.assertEqual(n_next, n_enc - 1)


class T2_DGsumm(unittest.TestCase):
    """Etape 2 : opérateurs display/filtrate/transformate/calculate/abstract."""
    def test_filtrate_removes_out_of_task_conditions(self):
        for p in PIDS:
            for n, d in PREP[p]["GS"].nodes(data=True):
                if d["type"] == "Problem":
                    self.assertIsNotNone(condition_theme(d["attrs"]["code"], d["attrs"]["libelle"]))

    def test_calculate_matches_independent_gold(self):
        """Dernière HbA1c, tendance et effectif calculés par DGsumm = valeurs calculées sur les tables brutes."""
        for p in PIDS:
            GS = PREP[p]["GS"]; a = GS.nodes["serie_HbA1c"]["attrs"]
            q = {x["qid"]: x for x in QS[p]}
            self.assertAlmostEqual(a["last"], q["Q2"]["gold"]["value"])
            self.assertEqual(a["last_date"], q["Q2"]["gold"]["date"])
            self.assertEqual(a["n2y"], q["Q10"]["gold"]["count"])
            if "Q3" in q:
                self.assertEqual(a["trend3y"], q["Q3"]["gold"]["trend"])

    def test_summary_is_compact(self):
        for p in PIDS:
            self.assertLess(PREP[p]["GS"].number_of_nodes(), 0.2 * PREP[p]["G"].number_of_nodes())


class T3_RDF_HERSE(unittest.TestCase):
    """Etape 3 : RDF-GraphSyn produit des nœuds anonymes, HERSE les élimine sans perte."""
    def test_no_blank_node_after_herse(self):
        for p in PIDS:
            self.assertGreater(R.stats(PREP[p]["rdf"])["blank"], 0)
            self.assertEqual(R.stats(PREP[p]["rdf_star"])["blank"], 0)

    def test_herse_reduces_size_and_tokens(self):
        for p in PIDS:
            P = PREP[p]
            self.assertLessEqual(len(P["rdf_star"]), len(P["rdf"]))
            self.assertLess(sum(i["ntok"] for i in P["items"]), sum(i["ntok_raw"] for i in P["items"]))

    def test_herse_preserves_values(self):
        for p in PIDS:
            KS = SummaryKB(PREP[p]["rdf_star"])
            a = PREP[p]["GS"].nodes["serie_HbA1c"]["attrs"]
            self.assertAlmostEqual(KS.series["4548-4"]["last"], round(a["last"], 2))
            self.assertEqual(KS.series["4548-4"]["last_date"], a["last_date"])


class T4_BuildContext(unittest.TestCase):
    """Etape 4 : budget respecté, contexte suffisant, repli pour les questions hors périmètre."""
    def test_budget_is_respected(self):
        for p in PIDS[:5]:
            for q in QS[p]:
                for B in (250, 1000):
                    for c, (ctx, used) in contexts(PREP[p], q["question"], B).items():
                        self.assertLessEqual(used, B, c)
                        self.assertEqual(used, sum(i["ntok"] for i in ctx))

    def test_in_scope_questions_are_sufficient(self):
        for p in PIDS:
            for q in QS[p]:
                if q["evidence"] and not q["qid"].startswith("QX"):
                    ctx, _ = contexts(PREP[p], q["question"], 500, conds=["OURS"])["OURS"]
                    self.assertTrue(C.sufficient(q["evidence"], C.covered(ctx))[0], q["qid"])

    def test_fallback_answers_out_of_scope(self):
        for p in PIDS:
            for q in QS[p]:
                if q["qid"].startswith("QX") and q["evidence"]:
                    cx = contexts(PREP[p], q["question"], 500, conds=["OURS", "OURS-GSonly"])
                    self.assertTrue(C.sufficient(q["evidence"], C.covered(cx["OURS"][0]))[0])
                    self.assertFalse(C.sufficient(q["evidence"], C.covered(cx["OURS-GSonly"][0]))[0])


class T5_VerifyAndExplain(unittest.TestCase):
    """Algorithme 2 : trois classes, monde fermé local, vérification hiérarchique."""
    @classmethod
    def setUpClass(cls):
        cls.X = Extractor(extra_meds=["acetaminophen", "ibuprofen", "naproxen sodium", "sodium fluoride"])

    def _status(self, p, text, V):
        P = PREP[p]
        res = verify(self.X.extract(text), SummaryKB(P["rdf_star"]), RecordKB(P["G"]), V)
        return decide(res)[0], res

    def test_reference_answers_are_ok(self):
        for p in PIDS:
            for q in QS[p]:
                if q["qid"].startswith("QX"):
                    continue
                for ref in q["references"]:
                    st, res = self._status(p, ref, "V4")
                    self.assertEqual(st, "OK", (ref, [(c, l) for c, l, _ in res]))

    def test_injected_errors_are_ko(self):
        rng = random.Random(0)
        for p in PIDS:
            for q in QS[p]:
                if q["qid"].startswith("QX"):
                    continue
                for pt, text in perturbations(q, REC[p], rng):
                    st, _ = self._status(p, text, "V4")
                    if pt in ("P1", "P2", "P3", "P4", "P5"):
                        self.assertEqual(st, "KO", (pt, text))
                    else:
                        self.assertEqual(st, "OK", (pt, text))

    def test_circularity_of_summary_only_verification(self):
        """Un fait vrai écarté par le résumé est rejeté par V3 mais accepté par V4."""
        found = False
        for p in PIDS:
            P = PREP[p]
            conds = set(REC[p]["conditions"].CODE)
            if "271737000" in conds:  # anémie : vraie mais hors tâche
                t = "La patiente présente par ailleurs : anémie."
                self.assertEqual(self._status(p, t, "V4")[0], "OK")
                self.assertNotEqual(self._status(p, t, "V3")[0], "OK")
                found = True
        self.assertTrue(found)

    def test_witness_subgraph_is_citable(self):
        p = PIDS[0]; q = [x for x in QS[p] if x["qid"] == "Q2"][0]
        _, res = self._status(p, q["references"][0], "V4")
        W = [w for _, l, ws in res for w in ws]
        self.assertTrue(W and all(t[0].startswith("ex:") for t in W))


if __name__ == "__main__":
    unittest.main(verbosity=2)
