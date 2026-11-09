"""Unit tests for scoped completeness statements and guarantee-aware verification (RQ4)."""
import os, random, sys, unittest
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.completeness import Statements, guard, SUPO
from ehrkg.verify import SUP, CON, UNV

REF = "2026-06-30"
def kb():
    return SimpleNamespace(ref=REF,
                           obs={"4548-4": [("2024-03-01", 7.4, "O1"), ("2025-02-01", 7.1, "O2"), ("2025-11-01", 6.9, "O3")]},
                           drugs={"metformine": [("2016-01-01", None, "M1")]},
                           conds={"44054006": [("2010-05-01", "C1")]})

class TestStatements(unittest.TestCase):
    def test_complete_window(self):
        S = Statements({(("OBS", "4548-4"), 2025)}, REF)
        self.assertFalse(S.complete(("OBS", "4548-4"), 2024))
        self.assertTrue(S.complete(("OBS", "4548-4"), 2026))
        self.assertTrue(S.complete(("MED",), 2000))
        self.assertFalse(S.complete_upto(("OBS", "4548-4")))

    def test_declarations(self):
        removed = [(("MED",), 2025), (("OBS", "x"), 2020)]
        self.assertEqual(Statements.declare(removed, REF, "naive").inc, set())
        self.assertEqual(len(Statements.declare(removed, REF, "accurate").inc), 2)
        w = Statements.declare(removed, REF, "wrong", rng=random.Random(0), wrong_rate=1.0)
        self.assertEqual(w.inc, set())

class TestGuard(unittest.TestCase):
    def test_last_value_conflict_needs_guarantee(self):
        c = ("VALUE", "4548-4", 6.5, "2026-04-01", True)        # claims a later HbA1c than recorded
        self.assertEqual(guard(c, CON, kb(), Statements({(("OBS", "4548-4"), 2026)}, REF)), UNV)
        self.assertEqual(guard(c, CON, kb(), Statements(set(), REF)), CON)

    def test_earlier_last_claim_is_soundly_contradicted(self):
        c = ("VALUE", "4548-4", 7.1, "2025-02-01", True)        # a later value is recorded
        self.assertEqual(guard(c, CON, kb(), Statements({(("OBS", "4548-4"), 2025)}, REF)), CON)

    def test_support_of_last_value_is_qualified(self):
        c = ("VALUE", "4548-4", 6.9, "2025-11-01", True)
        self.assertEqual(guard(c, SUP, kb(), Statements({(("OBS", "4548-4"), 2026)}, REF)), SUPO)
        self.assertEqual(guard(c, SUP, kb(), Statements(set(), REF)), SUP)

    def test_count_lower_bound(self):
        S = Statements({(("OBS", "4548-4"), 2025)}, REF)
        self.assertEqual(guard(("COUNT", "4548-4", 1), CON, kb(), S), CON)    # 2 recorded > 1 claimed
        self.assertEqual(guard(("COUNT", "4548-4", 4), CON, kb(), S), UNV)    # more may be missing

    def test_absent_drug_and_negation(self):
        S = Statements({(("MED",), 2026)}, REF)
        self.assertEqual(guard(("DRUG", "sitagliptine", True), CON, kb(), S), UNV)
        self.assertEqual(guard(("DRUG", "sitagliptine", True), CON, kb(), Statements(set(), REF)), CON)
        self.assertEqual(guard(("NEG", "statine"), SUP, kb(), S), UNV)
        self.assertEqual(guard(("DRUG", "metformine", True), SUP, kb(), S), SUP)   # existence: sound

    def test_diagnosis_date(self):
        S = Statements({(("COND",), 2008)}, REF)
        self.assertEqual(guard(("DXDATE", "44054006", "2012-01-01"), CON, kb(), S), CON)  # later than recorded
        self.assertEqual(guard(("DXDATE", "44054006", "2008-01-01"), CON, kb(), S), UNV)  # earlier fact may hide

if __name__ == "__main__":
    unittest.main()
