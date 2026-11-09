"""Step 4: BuildContext for every experimental condition.

A context is a list of items; each item has a text and the set of fact ids it
states (`covers`). `pack` fills a token budget. Conditions:
  TEXT-FULL  concatenated clinical notes (most recent kept under the budget)
  TEXT-RAG   BM25 over note chunks (query expanded by the shared linker)
  KG-DG      entity-linked retrieval of raw facts in the full data graph DG
             (= ablation "sans DGsumm"; generic graph retrieval)
  KG-PPR     HippoRAG-style retrieval: personalized PageRank on DG seeded by
             the concept nodes linked to the question (Gutierrez et al., 2024)
  OURS       G*_RDF items ranked by question link and profile UP
  OURS-noHERSE   same items, linearised from G_RDF (blank nodes)
  OURS-noUP      uniform profile
"""
import math, re
from collections import Counter
from .concepts import OBS, COMPLICATIONS, link_question, med_ingredient, condition_theme
from . import rdf as R

TOK = re.compile(r"\w+|[^\w\s]")
def ntok(s):
    """Approximate token count (word/punctuation segmentation)."""
    return len(TOK.findall(s))

def fmtv(v):
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))
    return str(v)

# ---------------------------------------------------------------- clinical notes
def render_notes(G):
    """One note per encounter: list of (date, [(fragment, fid|None)])."""
    by_enc = {}
    for n, d in G.nodes(data=True):
        if d.get("kind") in ("Observation", "Condition", "Medication", "Procedure"):
            enc = [v for _, v, k in G.out_edges(n, data="label") if k == "recordedIn"]
            if enc:
                by_enc.setdefault(enc[0], []).append((n, d))
    notes = []
    for e, ed in G.nodes(data=True):
        if ed.get("kind") != "Encounter":
            continue
        fr = [(f"Consultation du {ed['date']} ({ed['cls']}) : {ed['desc']}."
               + (f" Motif : {ed['reason']}." if ed.get("reason") else ""), e)]
        items = by_enc.get(e, [])
        for kind, title in [("Condition", "Diagnostic"), ("Observation", "Mesure"),
                            ("Medication", "Prescription"), ("Procedure", "Acte")]:
            for n, d in sorted((x for x in items if x[1]["kind"] == kind), key=lambda x: x[0]):
                if kind == "Condition":
                    t = f"{title} : {d['label']} (début {d['start']}" + (f", fin {d['stop']})." if d["stop"] else ").")
                elif kind == "Observation":
                    t = f"{title} : {d['label']} = {fmtv(d['value'])} {d['unit']}."
                elif kind == "Medication":
                    t = (f"{title} : {d['label']} (du {d['start']}, "
                         + (f"jusqu'au {d['stop']})." if d["stop"] else "sans date de fin)."))
                else:
                    t = f"{title} : {d['label']}."
                fr.append((t, n))
        notes.append((ed["date"], e, fr))
    notes.sort()
    return notes

def note_items(notes, max_tok=350):
    """Chunks of notes (split on fragment boundaries)."""
    items = []
    for date, e, fr in notes:
        cur, cov, nt = [], set(), 0
        for t, f in fr:
            k = ntok(t)
            if cur and nt + k > max_tok:
                items.append(dict(text=" ".join(cur), covers=cov, date=date)); cur, cov, nt = [], set(), 0
                cur.append(f"(suite {date})"); nt = 3
            cur.append(t); cov.add(f); nt += k
        if cur:
            items.append(dict(text=" ".join(cur), covers=cov, date=date))
    for it in items:
        it["ntok"] = ntok(it["text"])
    return items

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [[w.lower() for w in re.findall(r"\w+", d)] for d in docs]
        self.N = len(docs); self.avg = sum(map(len, self.toks)) / max(1, self.N)
        self.df = Counter(w for t in self.toks for w in set(t)); self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in self.toks]
    def scores(self, q):
        q = [w.lower() for w in re.findall(r"\w+", q)]
        out = []
        for i, tf in enumerate(self.tf):
            s, L = 0.0, len(self.toks[i])
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out

def pack(items, budget, order=None):
    """Greedy fill of the budget following `order` (indices); skip items that do not fit."""
    order = order if order is not None else range(len(items))
    sel, used = [], 0
    for i in order:
        if used + items[i]["ntok"] <= budget:
            sel.append(i); used += items[i]["ntok"]
    return sel, used

def ctx_text_full(note_its, budget):
    order = list(range(len(note_its)))[::-1]          # most recent first
    sel, used = pack(note_its, budget, order)
    sel.sort()
    return [note_its[i] for i in sel], used

def ctx_text_rag(note_its, bm25, question, budget):
    L = link_question(question)
    q = question + " " + " ".join(L["en"])
    sc = bm25.scores(q)
    order = sorted(range(len(note_its)), key=lambda i: (-sc[i], -i))
    sel, used = pack(note_its, budget, order)
    sel.sort()
    return [note_its[i] for i in sel], used

# ---------------------------------------------------------------- KG over DG
def dg_items(G):
    its = []
    for n, d in G.nodes(data=True):
        k = d.get("kind")
        if k == "Observation":
            t = f"{d['date']} | mesure | {d['label']} = {fmtv(d['value'])} {d['unit']} | [{n}]"
            its.append(dict(text=t, covers={n}, date=d["date"], kind=k, code=d["code"]))
        elif k == "Condition":
            t = f"{d['start']} | diagnostic | {d['label']}" + (f" (fin {d['stop']})" if d["stop"] else "") + f" | [{n}]"
            its.append(dict(text=t, covers={n}, date=d["start"], kind=k, code=d["code"], label=d["label"]))
        elif k == "Medication":
            t = (f"{d['start']} | prescription | {d['label']} | fin: {d['stop'] or 'aucune'} | [{n}]")
            mi = med_ingredient(d["label"])
            its.append(dict(text=t, covers={n}, date=d["start"], kind=k, medclass=mi[1] if mi else None))
    for it in its:
        it["ntok"] = ntok(it["text"])
    return its

def ctx_kg_dg(items, question, budget):
    L = link_question(question)
    def rel(it):
        if it["kind"] == "Observation" and it["code"] in L["obs"]: return 1
        if it["kind"] == "Condition" and (it["code"] in L["cond"] or
                                          condition_theme(it["code"], it["label"]) in L["themes"]): return 1
        if it["kind"] == "Medication" and it["medclass"] in L["medclass"]: return 1
        return 0
    # relevant facts only (most recent first); if nothing is linked, most recent facts
    rel_idx = sorted([i for i in range(len(items)) if rel(items[i])], key=lambda i: items[i]["date"], reverse=True)
    if not rel_idx:
        rel_idx = sorted(range(len(items)), key=lambda i: items[i]["date"], reverse=True)
    sel, used = pack(items, budget, rel_idx)
    sel.sort(key=lambda i: items[i]["date"])
    return [items[i] for i in sel], used

# ---------------------------------------------------------------- HippoRAG-style PPR over DG
def ppr_seeds(G, L):
    """Concept nodes of DG linked to the question (HippoRAG: query entities -> graph nodes)."""
    seeds = set()
    for n, d in G.nodes(data=True):
        if d.get("kind") != "Concept":
            continue
        if d["system"] == "LOINC" and d["code"] in L["obs"]: seeds.add(n)
        elif d["system"] == "SNOMED" and d["code"] in L["cond"]: seeds.add(n)
        elif d["system"] == "RxNorm":
            mi = med_ingredient(d["label"] or "")
            if mi and mi[1] in L["medclass"]: seeds.add(n)
    return seeds

def ppr_scores(G, UG, question, alpha=0.5):
    """Personalized PageRank (damping 0.5 as in HippoRAG) restarting on the linked concepts;
    the patient node is the restart node when nothing is linked."""
    import networkx as nx
    seeds = ppr_seeds(G, link_question(question)) or {G.graph["patient"]}
    pers = {n: 1.0 / len(seeds) for n in seeds}
    return nx.pagerank(UG, alpha=alpha, personalization=pers, tol=1e-8, max_iter=200)

def ctx_kg_ppr(items, scores, budget):
    """Facts of DG ranked by PPR score; among equal scores, most recent first."""
    order = sorted(range(len(items)), key=lambda i: items[i]["date"], reverse=True)
    order.sort(key=lambda i: -scores.get(next(iter(items[i]["covers"])), 0.0))   # stable sort
    sel, used = pack(items, budget, order)
    sel.sort(key=lambda i: items[i]["date"])
    return [items[i] for i in sel], used


# ---------------------------------------------------------------- explicit serialisations (S2, S3)
FULL_LABEL = {"4548-4": "hémoglobine glyquée (HbA1c)", "2339-0": "glycémie", "33914-3": "débit de filtration glomérulaire estimé (DFG)",
              "38483-4": "créatinine", "14959-1": "microalbuminurie", "8480-6": "pression artérielle systolique",
              "8462-4": "pression artérielle diastolique", "18262-6": "LDL-cholestérol", "2093-3": "cholestérol total",
              "2085-9": "HDL-cholestérol", "2571-8": "triglycérides", "39156-5": "indice de masse corporelle (IMC)"}
NG = " [non garanti : données possiblement incomplètes]"

def render_explicit(it, guarantees=None):
    """S2: one attribute per line, full labels, explicit units and dates.
    S3 (guarantees given): S2 + a marker on every value that is not guaranteed
    and the source fact id of last values (provenance)."""
    a, typ = it["attrs"], it["type"]
    g = guarantees if guarantees is not None else {}
    S3 = guarantees is not None
    def m(attr):
        return NG if (S3 and not g.get(attr, True)) else ""
    L = []
    if typ == "Patient":
        L.append(f"Patient : sexe {a['sexe']}, date de naissance {a['naissance']}.")
        L.append(f"- date de référence du dossier : {a['date_reference']}")
    elif typ == "Series":
        code = a["code"]; u = a["unite"]
        L.append(f"Série de mesures : {FULL_LABEL.get(code, a['analyte'])} (unité : {u})")
        src = f" [source : {a['last_fid']}]" if S3 and a.get("last_fid") else ""
        L.append(f"- dernière valeur mesurée : {fmtv(a['last'])} {u}, le {a['last_date']}{m('last')}{src}")
        L.append(f"- nombre de mesures au cours des 2 dernières années : {a['n2y']}{m('n2y')}")
        if a.get("trend3y"):
            L.append(f"- tendance sur les 3 dernières années : {a['trend3y']} ({fmtv(a['slope3y'])} {u} par an){m('trend3y')}")
        else:
            L.append("- tendance sur les 3 dernières années : non calculable (moins de 3 mesures)")
        L.append(f"- nombre total de mesures : {a['n']}, de {a['first_date']} à {a['last_date']}{m('n')}")
        L.append(f"- valeur minimale : {fmtv(a['min'])} {u} ; valeur maximale : {fmtv(a['max'])} {u}{m('min')}")
    elif typ == "Problem":
        L.append(f"Problème de santé : {a['libelle']}")
        L.append(f"- début : {a['debut']}{m('debut')}")
        if a.get("resolu"):
            L.append(f"- résolu le : {a['resolu']}")
    elif typ == "ComplicationGroup":
        org = a["organes"]
        if org:
            L.append("Complications du diabète documentées" + (" (liste possiblement incomplète)" if m("organes") else " (liste complète)") + " :")
            for o, labs in org.items():
                for lab, d in labs.items():
                    L.append(f"- {lab} (atteinte {o}), depuis le {d}")
        else:
            L.append("Complications du diabète documentées : aucune" + (NG if m("organes") else " (liste complète)"))
    elif typ == "DrugEpisode":
        L.append(f"Traitement : {a['principe']} (classe : {a['classe']})")
        L.append(f"- en cours à la date de référence : {'oui' if a['en_cours'] else 'non'}{m('en_cours')}")
        L.append(f"- première prescription : {a['debut']}{m('debut')}")
        L.append(f"- dernière prescription : {a['dernier']} ; nombre de prescriptions : {a['n_prescriptions']}{m('dernier')}")
    elif typ == "TreatmentSummary":
        cur = ", ".join(a["en_cours"]) or "aucun"
        hist = ", ".join(a["liste_complete"]) or "aucun"
        L.append("Antidiabétiques en cours" + (" (liste possiblement incomplète)" if m("en_cours") else " (liste complète)") + f" : {cur}")
        L.append("Antidiabétiques déjà prescrits" + (" (liste possiblement incomplète)" if m("liste_complete") else " (liste complète)") + f" : {hist}")
    elif typ == "Indicator":
        L.append(f"Indicateur : {a['indicateur']} = {a['valeur']}{m('valeur')}")
        L.append(f"- calculé à partir de : {a['base']}")
    return "\n".join(L)

S3_HEADER = ("Les valeurs suivies de « non garanti » ont été calculées sur des données possiblement incomplètes ; "
             "toutes les autres valeurs sont garanties complètes pour la période concernée.")

def fallback_chrono(items, stmts=None):
    """S2/S3 fallback: record facts in chronological order, most recent per concept marked;
    with statements (S3), the most recent fact is marked 'non garanti' when a later fact may be missing."""
    items = sorted(items, key=lambda it: it["date"])
    last = {}
    for i, it in enumerate(items):
        last[it.get("code") or it.get("label")] = i
    out = []
    for i, it in enumerate(items):
        is_last = last.get(it.get("code") or it.get("label")) == i
        t = it["text"] + (" (la plus récente)" if is_last else "")
        if is_last and stmts is not None and it.get("date"):
            y = int(it["date"][:4])
            ok = (stmts.complete(("OBS", it["code"]), y) if it.get("kind") == "Observation"
                  else stmts.complete_upto(("COND",)) if it.get("kind") == "Condition" else stmts.complete(("MED",), y))
            t += "" if ok else NG
        out.append(dict(it, text=t, ntok=ntok(t)))
    head = "Faits du dossier hors résumé (ordre chronologique, du plus ancien au plus récent) :"
    return [dict(text=head, covers=set(), ntok=ntok(head), date="")] + out

# ---------------------------------------------------------------- ours
def ours_items(GS, triples_star, iri_star, triples_raw, roots_raw):
    its = []
    for nid, d in GS.nodes(data=True):
        if nid == "patient":
            root = "ex:patient"
            ts = [t for t in triples_star if t[0] == root and not t[1].startswith("ehr:has")]
            txt = R.linearize_grouped(ts, root)
            txt_raw = R.linearize_triples([t for t in triples_raw if t[0] == root and not t[1].startswith("ehr:has")])
        else:
            root = iri_star[nid]
            txt = R.linearize_grouped(R.subtree(triples_star, root), root)
            txt_raw = R.linearize_triples([("ex:patient", f"ehr:has{d['type']}", roots_raw[nid])]
                                          + R.subtree(triples_raw, roots_raw[nid]))
        it = dict(id=nid, type=d["type"], theme=d["theme"], attrs=d["attrs"], covers=set(d["covers"]),
                  text=txt, ntok=ntok(txt), text_raw=txt_raw, ntok_raw=ntok(txt_raw))
        it["text_s2"] = render_explicit(it); it["ntok_s2"] = ntok(it["text_s2"])
        it["text_s3"] = render_explicit(it, d.get("guaranteed", {})); it["ntok_s3"] = ntok(it["text_s3"])
        its.append(it)
    return its

def ours_relevance(it, L):
    a = it["attrs"]
    if it["type"] == "Patient": return 2.0
    if it["type"] == "Series" and a["code"] in L["obs"]: return 1.0
    if it["type"] == "Problem" and a["code"] in L["cond"]: return 1.0
    if it["type"] == "ComplicationGroup" and L["cond"] & set(COMPLICATIONS): return 1.0
    if it["type"] in ("DrugEpisode",) and a["classe"] in L["medclass"]: return 1.0
    if it["type"] == "TreatmentSummary" and "antidiabetique" in L["medclass"]: return 1.0
    if it["type"] == "Indicator" and it["theme"] in L["themes"]: return 0.8
    if it["theme"] in L["themes"]: return 0.5
    return 0.1

def ctx_ours(items, question, budget, profile_w, use_herse=True, dg_items=None, min_score=0.5, serial="S1", stmts=None):
    """G*_RDF items ranked by relevance x profile weight. For a question, only
    items with relevance >= min_score are kept. If the question is linked to
    concepts that the summary does not contain and `dg_items` is given, the
    remaining budget is filled by record-level retrieval (fallback to DG)."""
    L = link_question(question) if question else dict(obs=set(), cond=set(), medclass=set(), themes=set(), en=[])
    if serial in ("S2", "S3"):
        k = serial.lower()
        view = [dict(it, ntok=it["ntok_" + k], text=it["text_" + k]) for it in items]
    else:
        key = "ntok" if use_herse else "ntok_raw"
        view = [dict(it, ntok=it[key], text=it["text"] if use_herse else it["text_raw"]) for it in items]
    def score(it):
        w = profile_w.get(it["theme"], 1.0) if it["theme"] else 1.0
        return ours_relevance(it, L) * w
    order = sorted(range(len(view)), key=lambda i: (-score(view[i]), view[i]["ntok"]))
    if question:
        order = [i for i in order if ours_relevance(view[i], L) >= min_score]
    if serial == "S3":
        hdr = dict(text=S3_HEADER, covers=set(), ntok=ntok(S3_HEADER))
        budget -= hdr["ntok"]
    sel, used = pack(view, budget, order)
    sel.sort(key=lambda i: order.index(i))
    ctx = [view[i] for i in sel]
    if serial == "S3":
        ctx = [hdr] + ctx; used += hdr["ntok"]; budget += hdr["ntok"]
    if question and dg_items is not None:
        in_gs_obs = {it["attrs"]["code"] for it in items if it["type"] == "Series"}
        in_gs_cond = {it["attrs"]["code"] for it in items if it["type"] == "Problem"} | \
                     ({c for c in COMPLICATIONS} if any(it["type"] == "ComplicationGroup" for it in items) else set())
        miss_obs, miss_cond = L["obs"] - in_gs_obs, L["cond"] - in_gs_cond
        if miss_obs or miss_cond:
            cand = [it for it in dg_items if (it["kind"] == "Observation" and it["code"] in miss_obs)
                    or (it["kind"] == "Condition" and it["code"] in miss_cond)]
            cand.sort(key=lambda it: it["date"], reverse=True)
            sel2, used2 = pack(cand, budget - used - (20 if serial in ("S2", "S3") else 0))
            if serial in ("S2", "S3") and sel2:
                fb = fallback_chrono([cand[i] for i in sel2], stmts if serial == "S3" else None)
                ctx += fb; used += sum(x["ntok"] for x in fb)
            else:
                ctx += [cand[i] for i in sel2]; used += used2
    return ctx, used

def covered(ctx):
    s = set()
    for it in ctx:
        s |= it["covers"]
    return s

def sufficient(evidence, cov):
    """evidence: list of alternative sets; returns (sufficient, best recall)."""
    if not evidence:
        return None, None
    best = max(len(e & cov) / len(e) if e else 1.0 for e in evidence)
    return any(e <= cov for e in evidence), best

DERIVED = ("S:", "CX:", "TX:")
def raw_evidence(evidence):
    """Evidence sets made of record facts only (no derived fact computed by DGsumm)."""
    return [e for e in evidence if not any(str(x).startswith(DERIVED) for x in e)]

def to_text(ctx):
    return "\n".join(it["text"] for it in ctx)
