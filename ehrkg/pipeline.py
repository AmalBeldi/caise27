"""Algorithm 1 end to end (symbolic part) + construction of every condition's context."""
import time
from .dg import build_dg
from .dgsumm import dgsumm, PROFILES
from . import rdf as R
from . import context as C

CONDITIONS = ["TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS", "OURS-GSonly", "OURS-noHERSE", "OURS-noUP"]
# serialisation study (CAiSE): OURS = S1 (grouped), OURS-S2 explicit, OURS-S3 explicit + guarantees/provenance
SERIAL_CONDITIONS = ["OURS-S2", "OURS-S3"]

def prepare(r, profile="generaliste", statements=None):
    """statements: completeness statements (ehrkg.completeness.Statements); None = record declared complete."""
    from .completeness import Statements, annotate
    T = {}
    t = time.perf_counter(); G = build_dg(r); T["BuildDG"] = time.perf_counter() - t
    t = time.perf_counter(); GS = dgsumm(G, profile); T["DGsumm"] = time.perf_counter() - t
    GSu = dgsumm(G, "uniforme")
    S = statements if statements is not None else Statements(set(), G.graph["ref_date"])
    annotate(GS, S); annotate(GSu, S)
    t = time.perf_counter(); tr, roots = R.rdf_graphsyn(GS); T["RDF-GraphSyn"] = time.perf_counter() - t
    t = time.perf_counter(); ts, iri = R.herse(tr, roots); T["HERSE"] = time.perf_counter() - t
    tru, rootsu = R.rdf_graphsyn(GSu); tsu, iriu = R.herse(tru, rootsu)
    t = time.perf_counter()
    items = C.ours_items(GS, ts, iri, tr, roots)
    items_u = C.ours_items(GSu, tsu, iriu, tru, rootsu)
    notes = C.render_notes(G); nits = C.note_items(notes)
    bm = C.BM25([x["text"] for x in nits]); dits = C.dg_items(G)
    import networkx as nx
    UG = nx.Graph(G)                     # undirected simple view of DG for PPR
    T["BuildContext-index"] = time.perf_counter() - t
    return dict(G=G, GS=GS, rdf=tr, roots=roots, rdf_star=ts, iri=iri, items=items, items_u=items_u,
                notes=nits, bm25=bm, dg_items=dits, UG=UG, ppr_cache={}, times=T, profile=profile, stmts=S)

def contexts(P, question, budget, conds=CONDITIONS):
    w = PROFILES[P["profile"]]
    out = {}
    for c in conds:
        if c == "TEXT-FULL": out[c] = C.ctx_text_full(P["notes"], budget)
        elif c == "TEXT-RAG": out[c] = C.ctx_text_rag(P["notes"], P["bm25"], question or "suivi du diabète de type 2", budget)
        elif c == "KG-DG": out[c] = C.ctx_kg_dg(P["dg_items"], question or "", budget)
        elif c == "KG-PPR":
            q = question or "suivi du diabète de type 2"
            if q not in P["ppr_cache"]:
                P["ppr_cache"][q] = C.ppr_scores(P["G"], P["UG"], q)
            out[c] = C.ctx_kg_ppr(P["dg_items"], P["ppr_cache"][q], budget)
        elif c == "OURS": out[c] = C.ctx_ours(P["items"], question, budget, w, dg_items=P["dg_items"])
        elif c == "OURS-GSonly": out[c] = C.ctx_ours(P["items"], question, budget, w)
        elif c == "OURS-noHERSE": out[c] = C.ctx_ours(P["items"], question, budget, w, use_herse=False,
                                                      dg_items=P["dg_items"])
        elif c == "OURS-noUP": out[c] = C.ctx_ours(P["items_u"], question, budget, PROFILES["uniforme"],
                                                   dg_items=P["dg_items"])
        elif c in ("OURS-S2", "OURS-S3"):
            out[c] = C.ctx_ours(P["items"], question, budget, w, dg_items=P["dg_items"], serial=c[-2:], stmts=P["stmts"])
    return out
