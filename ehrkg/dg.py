"""Step 1: BuildDG — projection of an EHR into a labelled data graph DG=(V,E,lambda).

Node kinds: Patient, Encounter, Observation, Condition, Medication, Procedure,
Concept. Edges: hasEncounter, NEXT (temporal order of encounters), recordedIn
(fact -> encounter), about (patient -> fact), instanceOf (fact -> concept).
Every fact node keeps its stable fact id (`fid`) for provenance.
"""
import networkx as nx

def build_dg(r):
    G = nx.MultiDiGraph()
    pid = r["patient"]["id"]
    P = "P:" + pid[-8:]
    G.add_node(P, kind="Patient", **{k: v for k, v in r["patient"].items() if k != "id"})
    enc = r["encounters"].sort_values("START")
    prev = None
    for e in enc.itertuples():
        G.add_node(e.fid, kind="Encounter", date=e.START, cls=e.ENCOUNTERCLASS,
                   desc=e.DESCRIPTION, reason=e.REASONDESCRIPTION if isinstance(e.REASONDESCRIPTION, str) else None)
        G.add_edge(P, e.fid, label="hasEncounter")
        if prev:
            G.add_edge(prev, e.fid, label="NEXT")
        prev = e.fid
    encmap = dict(zip(r["encounters"].Id, r["encounters"].fid))

    def concept(system, code, desc):
        c = f"{system}:{code}"
        if c not in G:
            G.add_node(c, kind="Concept", system=system, code=code, label=desc)
        return c

    for o in r["observations"].itertuples():
        G.add_node(o.fid, kind="Observation", code=o.CODE, label=o.DESCRIPTION, date=o.DATE,
                   value=o.num if o.num == o.num else o.VALUE, unit=o.UNITS if isinstance(o.UNITS, str) else "")
        G.add_edge(o.fid, concept("LOINC", o.CODE, o.DESCRIPTION), label="instanceOf")
        G.add_edge(P, o.fid, label="about")
        if o.ENCOUNTER in encmap:
            G.add_edge(o.fid, encmap[o.ENCOUNTER], label="recordedIn")
    for c in r["conditions"].itertuples():
        G.add_node(c.fid, kind="Condition", code=c.CODE, label=c.DESCRIPTION, start=c.START,
                   stop=c.STOP if isinstance(c.STOP, str) else None)
        G.add_edge(c.fid, concept("SNOMED", c.CODE, c.DESCRIPTION), label="instanceOf")
        G.add_edge(P, c.fid, label="about")
        if c.ENCOUNTER in encmap:
            G.add_edge(c.fid, encmap[c.ENCOUNTER], label="recordedIn")
    for m in r["medications"].itertuples():
        G.add_node(m.fid, kind="Medication", code=m.CODE, label=m.DESCRIPTION, start=m.START,
                   stop=m.STOP if isinstance(m.STOP, str) else None,
                   reason=m.REASONDESCRIPTION if isinstance(m.REASONDESCRIPTION, str) else None)
        G.add_edge(m.fid, concept("RxNorm", m.CODE, m.DESCRIPTION), label="instanceOf")
        G.add_edge(P, m.fid, label="about")
        if m.ENCOUNTER in encmap:
            G.add_edge(m.fid, encmap[m.ENCOUNTER], label="recordedIn")
    for p in r["procedures"].itertuples():
        G.add_node(p.fid, kind="Procedure", code=p.CODE, label=p.DESCRIPTION, date=p.START)
        G.add_edge(p.fid, concept("SNOMED", p.CODE, p.DESCRIPTION), label="instanceOf")
        G.add_edge(P, p.fid, label="about")
        if p.ENCOUNTER in encmap:
            G.add_edge(p.fid, encmap[p.ENCOUNTER], label="recordedIn")
    G.graph["patient"] = P
    G.graph["ref_date"] = r["patient"]["ref_date"]
    return G

def facts(G, kind):
    return [(n, d) for n, d in G.nodes(data=True) if d.get("kind") == kind]
