"""Tests of the generic core (gsum) on a tiny event record."""
import os, sys, unittest
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsum.core import Record, Statements, check, SUP, SUPO, CON, UNV

T = pd.Timestamp
R = Record("c1", [dict(id="e1", prop="O", t=T("2016-01-01"), value=5000.0),
                  dict(id="e2", prop="O", t=T("2016-01-10"), value=8000.0),
                  dict(id="e3", prop="A_Pending", t=T("2016-01-12"), value=1.0)], T("2016-01-20"))
GAP = Statements({("O", T("2016-01-15")), ("A_Pending", T("2016-01-15")), ("A_Cancelled", T("2016-01-15"))})

class TestCore(unittest.TestCase):
    def test_last_value(self):
        c = ("LAST", "O", 8000.0, None)
        self.assertEqual(check(c, R, Statements(), "guarantees")[0], SUP)
        self.assertEqual(check(c, R, GAP, "guarantees")[0], SUPO)
        stale = ("LAST", "O", 5000.0, T("2016-01-01"))
        self.assertEqual(check(stale, R, GAP, "guarantees")[0], CON)      # a later offer is recorded
        later = ("LAST", "O", 9000.0, T("2016-01-15"))
        self.assertEqual(check(later, R, GAP, "guarantees")[0], UNV)
        self.assertEqual(check(later, R, GAP, "assumed")[0], CON)

    def test_count_and_state(self):
        self.assertEqual(check(("COUNT", "O", 1, 30), R, GAP, "guarantees")[0], CON)   # 2 recorded: lower bound
        self.assertEqual(check(("COUNT", "O", 3, 30), R, GAP, "guarantees")[0], UNV)
        st = ("A_Pending", "A_Cancelled")
        self.assertEqual(check(("STATE", "A_Cancelled"), R, GAP, "guarantees", st)[0], UNV)
        self.assertEqual(check(("STATE", "A_Cancelled"), R, Statements(), "guarantees", st)[0], CON)

    def test_existence(self):
        self.assertEqual(check(("NOT_EXISTS", "A_Denied"), R, Statements(), "guarantees")[0], SUP)
        self.assertEqual(check(("NOT_EXISTS", "O"), R, GAP, "guarantees")[0], CON)
        self.assertEqual(check(("EXISTS", "A_Denied"), R, Statements({("A_Denied", T("2016-01-05"))}), "guarantees")[0], UNV)

if __name__ == "__main__":
    unittest.main()
