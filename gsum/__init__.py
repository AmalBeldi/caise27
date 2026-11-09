"""Domain-independent core of the method (summaries with guarantees).

core.py   entity records, scoped completeness statements over time intervals, guarantee-preserving
          summary operators (last, count over a window, first, current state, existence) and the
          three-valued verifier with its three regimes (completeness assumed / guarantees / open world).
bpic.py   instantiation for process event logs (BPI Challenge 2017, loan applications).
The EHR instantiation (package ehrkg) implements the same rules with yearly statement cells.
"""
