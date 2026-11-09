"""Step 3: RDF-GraphSyn (GS -> G_RDF) and HERSE (G_RDF -> G*_RDF).

RDF-GraphSyn produces a TBox (classes, properties) and an ABox. Following the
usual n-ary relation pattern, measures, periods, ranges and nested groups are
encoded with blank nodes.  HERSE (re-implementation of its principles for the
experiments) handles blank nodes by
  (1) skolemisation of blank nodes that denote a summary entity into stable
      IRIs (needed so that a proof sub-graph G_exp can cite them),
  (2) elimination of single-use n-ary blank nodes by lifting their properties
      into the parent with a composed property, and merging value+unit into
      one typed literal,
  (3) removal of duplicate triples.
Triples are plain python tuples; terms are strings ('ex:', 'ehr:', '_:b')
or python literals.
"""
import itertools, re

TYPES = ["Patient", "Series", "Problem", "ComplicationGroup", "DrugEpisode", "TreatmentSummary", "Indicator"]

def tbox():
    T = []
    for t in TYPES:
        T.append((f"ehr:{t}", "rdf:type", "owl:Class"))
    props = ["hasSeries", "hasProblem", "hasComplicationGroup", "hasDrugEpisode", "hasTreatmentSummary",
             "hasIndicator", "code", "label", "unit", "value", "date", "count", "countLast2Years",
             "min", "max", "direction", "slope", "lastMeasure", "range", "trend3y", "firstMeasureDate",
             "ingredient", "drugClass", "period", "start", "lastPrescription", "active", "nPrescriptions",
             "onset", "resolved", "organ", "complication", "complete", "currentItem", "historyItem",
             "indicator", "basis", "derivedFrom"]
    for p in props:
        T.append((f"ehr:{p}", "rdf:type", "owl:DatatypeProperty"))
    return T

class Store:
    def __init__(self):
        self.triples = []
        self._b = itertools.count(1)
    def bnode(self):
        return f"_:b{next(self._b)}"
    def add(self, s, p, o):
        self.triples.append((s, p, o))

def rdf_graphsyn(GS, with_provenance=True):
    """Naive translation GS -> G_RDF (blank-node based ABox) + subject index."""
    S = Store()
    P = "ex:patient"
    S.add(P, "rdf:type", "ehr:Patient")
    for k, v in GS.nodes["patient"]["attrs"].items():
        S.add(P, f"ehr:{k}", v)
    roots = {}  # summary node id -> blank root
    for nid, d in GS.nodes(data=True):
        if nid == "patient":
            continue
        a, t = d["attrs"], d["type"]
        b = S.bnode(); roots[nid] = b
        S.add(P, f"ehr:has{t}", b); S.add(b, "rdf:type", f"ehr:{t}")
        if t == "Series":
            S.add(b, "ehr:label", a["analyte"]); S.add(b, "ehr:code", a["code"])
            m = S.bnode(); S.add(b, "ehr:lastMeasure", m)
            S.add(m, "ehr:value", a["last"]); S.add(m, "ehr:unit", a["unite"]); S.add(m, "ehr:date", a["last_date"])
            S.add(b, "ehr:count", a["n"]); S.add(b, "ehr:countLast2Years", a["n2y"])
            S.add(b, "ehr:firstMeasureDate", a["first_date"])
            rg = S.bnode(); S.add(b, "ehr:range", rg); S.add(rg, "ehr:min", a["min"]); S.add(rg, "ehr:max", a["max"])
            if a["trend3y"]:
                tr = S.bnode(); S.add(b, "ehr:trend3y", tr)
                S.add(tr, "ehr:direction", a["trend3y"]); S.add(tr, "ehr:slope", round(a["slope3y"], 3))
        elif t == "Problem":
            S.add(b, "ehr:label", a["libelle"]); S.add(b, "ehr:code", a["code"]); S.add(b, "ehr:onset", a["debut"])
            if a["resolu"]:
                S.add(b, "ehr:resolved", a["resolu"])
        elif t == "ComplicationGroup":
            S.add(b, "ehr:complete", True)
            for organ, comps in a["organes"].items():
                og = S.bnode(); S.add(b, "ehr:organ", og); S.add(og, "ehr:label", organ)
                for lab, onset in comps.items():
                    c = S.bnode(); S.add(og, "ehr:complication", c)
                    S.add(c, "ehr:label", lab); S.add(c, "ehr:onset", onset)
        elif t == "DrugEpisode":
            S.add(b, "ehr:ingredient", a["principe"]); S.add(b, "ehr:drugClass", a["classe"])
            pe = S.bnode(); S.add(b, "ehr:period", pe)
            S.add(pe, "ehr:start", a["debut"]); S.add(pe, "ehr:lastPrescription", a["dernier"])
            S.add(b, "ehr:active", a["en_cours"]); S.add(b, "ehr:nPrescriptions", a["n_prescriptions"])
        elif t == "TreatmentSummary":
            S.add(b, "ehr:label", "traitements antidiabetiques"); S.add(b, "ehr:complete", True)
            for x in a["liste_complete"]:
                S.add(b, "ehr:historyItem", x)
            for x in a["en_cours"]:
                S.add(b, "ehr:currentItem", x)
        elif t == "Indicator":
            S.add(b, "ehr:indicator", a["indicateur"]); S.add(b, "ehr:value", a["valeur"]); S.add(b, "ehr:basis", a["base"])
        if with_provenance:
            for f in d["prov"]:
                S.add(b, "ehr:derivedFrom", f"dg:{f}")
    return S.triples, roots

def is_blank(x):
    return isinstance(x, str) and x.startswith("_:")

def herse(triples, roots):
    """Blank-node elimination. Returns (G*_RDF triples, root IRI map)."""
    out = {}
    iri = {b: f"ex:{nid}" for nid, b in roots.items()}  # (1) skolemisation of entity blank nodes
    by_s = {}
    for s, p, o in triples:
        by_s.setdefault(s, []).append((p, o))
    used_as_obj = {}
    for s, p, o in triples:
        if is_blank(o):
            used_as_obj[o] = used_as_obj.get(o, 0) + 1
    def lift(subject, p, o):
        """(2) lift a single-use n-ary blank node into its parent."""
        props = dict((pp.split(":")[1], oo) for pp, oo in by_s.get(o, []))
        pn = p.split(":")[1]
        if "value" in props and "unit" in props:                       # measure -> typed literal
            lit = f"{_fmt(props['value'])} {props['unit']}"
            if "date" in props:
                lit += f" @ {props['date']}"
            return [(subject, p, lit)]
        if set(props) <= {"start", "lastPrescription", "min", "max", "direction", "slope"}:
            return [(subject, f"ehr:{pn}_{k}", v) for k, v in props.items()]
        # nested group (complications): skolemise with a readable IRI
        lab = props.get("label", o[2:])
        node = f"{iri.get(subject, subject)}_{_slug(lab)}"
        res = [(subject, p, node)]
        for pp, oo in by_s.get(o, []):
            res += lift(node, pp, oo) if is_blank(oo) else [(node, pp, oo)]
        return res
    res = []
    for s, p, o in triples:
        if is_blank(s) and s not in iri:
            continue  # handled through its parent
        S_ = iri.get(s, s)
        if is_blank(o) and o in iri:
            res.append((S_, p, iri[o]))
        elif is_blank(o) and used_as_obj.get(o, 0) == 1:
            res += lift(S_, p, o)
        else:
            res.append((S_, p, o))
    seen, final = set(), []                                             # (3) dedup
    for t in res:
        k = tuple(str(x) for x in t)
        if k not in seen:
            seen.add(k); final.append(t)
    return final, {nid: iri[b] for nid, b in roots.items()}

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def stats(triples):
    nodes = set()
    for s, p, o in triples:
        nodes.add(s)
        if isinstance(o, str) and (o.startswith("_:") or o.startswith("ex:") or o.startswith("dg:")):
            nodes.add(o)
    return dict(triples=len(triples), blank=sum(1 for n in nodes if is_blank(n)), nodes=len(nodes))

# ---- linearisation ------------------------------------------------------------
def _short(x):
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, float):
        return _fmt(x)
    if isinstance(x, str):
        return x.replace("ehr:", "").replace("ex:", "")
    return str(x)

def subtree(triples, root):
    """All triples reachable from root through blank or ex: nested nodes (no provenance)."""
    by_s = {}
    for t in triples:
        by_s.setdefault(t[0], []).append(t)
    out, stack, seen = [], [root], set()
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        for t in by_s.get(s, []):
            if t[1] == "ehr:derivedFrom":
                continue
            out.append(t)
            o = t[2]
            if isinstance(o, str) and (is_blank(o) or (o.startswith("ex:") and o != root and o.startswith(root))):
                stack.append(o)
    return out

def linearize_triples(ts):
    """N-Triples-like linearisation (used without HERSE)."""
    return "\n".join(f"{_short(s)} {_short(p)} {_short(o)} ." for s, p, o in ts)

def linearize_grouped(ts, root):
    """Subject-grouped compact linearisation (used on G*_RDF)."""
    by = {}
    for s, p, o in ts:
        by.setdefault(s, []).append((p, o))
    lines = []
    for s, po in by.items():
        typ = [o for p, o in po if p == "rdf:type"]
        attrs = "; ".join(f"{_short(p)}={_short(o)}" for p, o in po if p != "rdf:type"
                          and not (isinstance(o, str) and o.startswith("ex:") and o in by))
        lines.append(f"[{_short(s)}]{(' ' + _short(typ[0])) if typ else ''}: {attrs}")
    return "\n".join(lines)
