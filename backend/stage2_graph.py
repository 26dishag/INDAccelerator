"""
Stage 2 — Knowledge graph extraction and construction.

Reads a cached Stage 1 paper object, runs Claude extraction across every
section and figure caption concurrently, normalises entities against a
synonym dictionary, builds a NetworkX directed graph, and serialises it
as Cytoscape-compatible JSON.
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import anthropic
import networkx as nx

# ---------------------------------------------------------------------------
# Entity normalisation — every surface form → canonical ID
# ---------------------------------------------------------------------------

ENTITY_SYNONYMS: dict[str, str] = {
    # Drug
    "11h": "DRUG_11H",
    "compound 11h": "DRUG_11H",

    # PDE4 isoforms
    "pde4": "ENZYME_PDE4",
    "pde4a": "ENZYME_PDE4A",
    "pde4a1": "ENZYME_PDE4A1",
    "pde4a10": "ENZYME_PDE4A10",
    "pde4b": "ENZYME_PDE4B",
    "pde4b2": "ENZYME_PDE4B2",
    "pde4c": "ENZYME_PDE4C",
    "pde4c1": "ENZYME_PDE4C1",
    "pde4d": "ENZYME_PDE4D",
    "pde4d2": "ENZYME_PDE4D2",
    "pde4d3": "ENZYME_PDE4D3",
    "phosphodiesterase 4": "ENZYME_PDE4",
    "phosphodiesterase 4b": "ENZYME_PDE4B",
    "phosphodiesterase 4d": "ENZYME_PDE4D",
    "phosphodiesterase-4": "ENZYME_PDE4",
    "pde4 inhibitor": "ENZYME_PDE4",

    # Signalling molecules
    "camp": "MOLECULE_CAMP",
    "cyclic amp": "MOLECULE_CAMP",
    "cyclic adenosine monophosphate": "MOLECULE_CAMP",
    "pka": "ENZYME_PKA",
    "protein kinase a": "ENZYME_PKA",
    "epac": "PROTEIN_EPAC",
    "epac1": "PROTEIN_EPAC",
    "epac2": "PROTEIN_EPAC",
    "creb": "TRANSCRIPTION_FACTOR_CREB",
    "camp response element-binding protein": "TRANSCRIPTION_FACTOR_CREB",
    "hsp20": "PROTEIN_HSP20",
    "heat shock protein 20": "PROTEIN_HSP20",

    # Inflammatory pathway
    "nf-kb": "TRANSCRIPTION_FACTOR_NFKB",
    "nf-κb": "TRANSCRIPTION_FACTOR_NFKB",
    "nfkb": "TRANSCRIPTION_FACTOR_NFKB",
    "nuclear factor kappa b": "TRANSCRIPTION_FACTOR_NFKB",
    "nuclear factor-κb": "TRANSCRIPTION_FACTOR_NFKB",
    "myd88": "PROTEIN_MYD88",
    "myeloid differentiation primary response 88": "PROTEIN_MYD88",
    "tlr4": "RECEPTOR_TLR4",
    "toll-like receptor 4": "RECEPTOR_TLR4",
    "il1r1": "RECEPTOR_IL1R1",
    "interleukin-1 receptor type 1": "RECEPTOR_IL1R1",
    "tradd": "GENE_TRADD",
    "fbl": "GENE_FBL",

    # Cytokines and inflammatory mediators
    "tnf-alpha": "CYTOKINE_TNFA",
    "tnf-α": "CYTOKINE_TNFA",
    "tnfa": "CYTOKINE_TNFA",
    "tnf": "CYTOKINE_TNFA",
    "tumor necrosis factor alpha": "CYTOKINE_TNFA",
    "tumor necrosis factor-alpha": "CYTOKINE_TNFA",
    "tumor necrosis factor-α": "CYTOKINE_TNFA",
    "il-6": "CYTOKINE_IL6",
    "il6": "CYTOKINE_IL6",
    "interleukin-6": "CYTOKINE_IL6",
    "interleukin 6": "CYTOKINE_IL6",
    "il-1beta": "CYTOKINE_IL1B",
    "il-1β": "CYTOKINE_IL1B",
    "il-1b": "CYTOKINE_IL1B",
    "il1b": "CYTOKINE_IL1B",
    "interleukin-1 beta": "CYTOKINE_IL1B",
    "interleukin-1β": "CYTOKINE_IL1B",
    "cxcl10": "CYTOKINE_CXCL10",
    "ip-10": "CYTOKINE_CXCL10",
    "ifn-gamma": "CYTOKINE_IFNG",
    "ifn-γ": "CYTOKINE_IFNG",
    "interferon gamma": "CYTOKINE_IFNG",
    "pge2": "LIPID_PGE2",
    "prostaglandin e2": "LIPID_PGE2",
    "nitric oxide": "MOLECULE_NO",
    "nitrites": "MOLECULE_NO",
    "no": "MOLECULE_NO",

    # Cell types
    "m1 macrophage": "CELL_M1_MACRO",
    "m1 macrophages": "CELL_M1_MACRO",
    "cd86+": "CELL_M1_MACRO",
    "m2 macrophage": "CELL_M2_MACRO",
    "m2 macrophages": "CELL_M2_MACRO",
    "cd206+": "CELL_M2_MACRO",
    "microglia": "CELL_MICROGLIA",
    "cd4+ t cell": "CELL_CD4T",
    "cd4+ t cells": "CELL_CD4T",
    "cd4 t cell": "CELL_CD4T",
    "cd8+ t cell": "CELL_CD8T",
    "cd8+ t cells": "CELL_CD8T",
    "pbmc": "CELL_PBMC",
    "peripheral blood mononuclear cell": "CELL_PBMC",
    "peripheral blood mononuclear cells": "CELL_PBMC",
    "raw264.7": "CELL_RAW2647",
    "raw 264.7": "CELL_RAW2647",
    "img": "CELL_IMG",
    "induced microglia-like": "CELL_IMG",
    "macrophage": "CELL_MACROPHAGE",
    "macrophages": "CELL_MACROPHAGE",

    # PK properties / measurements
    "brain penetration": "PROPERTY_BRAIN_PENETRATION",
    "blood-brain barrier": "PROPERTY_BBB",
    "bbb": "PROPERTY_BBB",
    "oral bioavailability": "PROPERTY_ORAL_BIOAVAILABILITY",
    "bioavailability": "PROPERTY_ORAL_BIOAVAILABILITY",
    "cmax": "MEASUREMENT_CMAX",
    "auc": "MEASUREMENT_AUC",
    "half-life": "MEASUREMENT_T12",
    "t1/2": "MEASUREMENT_T12",
    "clearance": "MEASUREMENT_CL",
    "volume of distribution": "MEASUREMENT_VD",
    "vd": "MEASUREMENT_VD",
    "tmax": "MEASUREMENT_TMAX",
    "t max": "MEASUREMENT_TMAX",
    "brain/plasma ratio": "MEASUREMENT_BRAIN_PLASMA_RATIO",

    # Safety endpoints
    "emesis": "OUTCOME_EMESIS",
    "vomiting": "OUTCOME_EMESIS",
    "nausea": "OUTCOME_NAUSEA",
    "herg": "TARGET_HERG",
    "herg channel": "TARGET_HERG",
    "cav1.2": "TARGET_CAV12",
    "nav1.5": "TARGET_NAV15",
    "cyp450": "ENZYME_CYP450",
    "cyp1a2": "ENZYME_CYP1A2",
    "cyp2c19": "ENZYME_CYP2C19",
    "cyp3a4": "ENZYME_CYP3A4",
    "cyp2d6": "ENZYME_CYP2D6",
    "mutagenicity": "OUTCOME_MUTAGENICITY",
    "genotoxicity": "OUTCOME_GENOTOXICITY",
    "cytotoxicity": "OUTCOME_CYTOTOXICITY",
    "maximum tolerated dose": "MEASUREMENT_MTD",
    "mtd": "MEASUREMENT_MTD",
    "body weight": "MEASUREMENT_BODY_WEIGHT",

    # Behavioural outcomes
    "anxiolytic": "OUTCOME_ANXIOLYTIC",
    "anxiolytic effect": "OUTCOME_ANXIOLYTIC",
    "antidepressant": "OUTCOME_ANTIDEPRESSANT",
    "antidepressant effect": "OUTCOME_ANTIDEPRESSANT",
    "elevated plus maze": "ASSAY_EPM",
    "epm": "ASSAY_EPM",
    "zero maze": "ASSAY_ZERO_MAZE",
    "forced swim test": "ASSAY_FST",
    "fst": "ASSAY_FST",
    "immobility": "MEASUREMENT_IMMOBILITY",

    # Biological processes
    "neuroinflammation": "PROCESS_NEUROINFLAMMATION",
    "neurodegeneration": "PROCESS_NEURODEGENERATION",
    "remyelination": "PROCESS_REMYELINATION",
    "apoptosis": "PROCESS_APOPTOSIS",
    "neuroprotection": "PROCESS_NEUROPROTECTION",
    "inflammation": "PROCESS_INFLAMMATION",
    "microglial activation": "PROCESS_MICROGLIAL_ACTIVATION",

    # Conditions / models
    "lps": "CONDITION_LPS",
    "lipopolysaccharide": "CONDITION_LPS",
    "lps mouse model": "MODEL_LPS_MOUSE",
    "lps model": "MODEL_LPS_MOUSE",

    # Comparator drugs
    "roflumilast": "DRUG_ROFLUMILAST",
    "rolipram": "DRUG_ROLIPRAM",
    "ibudilast": "DRUG_IBUDILAST",
    "rimonabant": "DRUG_RIMONABANT",
    "apremilast": "DRUG_APREMILAST",

    # Molecular / docking
    "pde4 catalytic domain": "PROTEIN_PDE4_CATALYTIC",
    "catalytic domain": "PROTEIN_PDE4_CATALYTIC",
    "binding site 2": "PROTEIN_PDE4_SITE2",
    "binding site 3": "PROTEIN_PDE4_SITE3",
    "site 2": "PROTEIN_PDE4_SITE2",
    "site 3": "PROTEIN_PDE4_SITE3",
    "glide score": "MEASUREMENT_GLIDE_SCORE",
    "docking score": "MEASUREMENT_GLIDE_SCORE",
}

NODE_TYPE_MAP: dict[str, str] = {
    "DRUG_": "drug",
    "ENZYME_": "enzyme",
    "RECEPTOR_": "receptor",
    "CYTOKINE_": "cytokine",
    "LIPID_": "lipid",
    "TRANSCRIPTION_FACTOR_": "transcription_factor",
    "PROTEIN_": "protein",
    "GENE_": "gene",
    "MOLECULE_": "molecule",
    "CELL_": "cell_type",
    "PROPERTY_": "property",
    "MEASUREMENT_": "measurement",
    "OUTCOME_": "outcome",
    "PROCESS_": "process",
    "CONDITION_": "condition",
    "MODEL_": "model",
    "ASSAY_": "assay",
    "TARGET_": "target",
}

# ---------------------------------------------------------------------------
# Claude extraction
# ---------------------------------------------------------------------------

EXTRACTION_TOOL = {
    "name": "extract_information",
    "description": (
        "Extract all biological, pharmacokinetic, safety, study design, and "
        "molecular information from this paper section"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["from_entity", "to_entity", "relationship", "category", "evidence_type", "source"],
                    "properties": {
                        "from_entity": {
                            "type": "string",
                            "description": "Subject entity exactly as named in the text",
                        },
                        "to_entity": {
                            "type": "string",
                            "description": "Object entity exactly as named in the text",
                        },
                        "relationship": {
                            "type": "string",
                            "description": (
                                "The relationship verb: inhibits, activates, reduces, increases, "
                                "measures, demonstrates, produces, requires, binds, "
                                "phosphorylates, degrades, etc"
                            ),
                        },
                        "relationship_direction": {
                            "type": "string",
                            "enum": ["activation", "inhibition", "binding", "measurement", "production", "neutral", "other"],
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "pathway_biology",
                                "pharmacokinetics",
                                "safety_tolerability",
                                "study_design",
                                "molecular_properties",
                            ],
                            "description": "Which of the five IND-relevant categories this relationship belongs to",
                        },
                        "evidence_type": {
                            "type": "string",
                            "enum": ["directly_measured", "paper_claims", "cited_from_reference"],
                            "description": (
                                "directly_measured: this section shows the experiment and result. "
                                "paper_claims: asserted as conclusion without showing raw data here. "
                                "cited_from_reference: attributed to a cited paper."
                            ),
                        },
                        "quantitative_value": {
                            "type": "string",
                            "description": "Any numerical result: IC50, Cmax, AUC, p-value, effect size, percentage, etc",
                        },
                        "experimental_conditions": {
                            "type": "object",
                            "properties": {
                                "assay_or_model": {"type": "string"},
                                "species": {
                                    "type": "string",
                                    "enum": ["mouse", "rat", "ferret", "dog", "nhp", "human", "in_vitro", "in_silico", "not_specified"],
                                },
                                "in_vivo_vitro": {
                                    "type": "string",
                                    "enum": ["in_vivo", "in_vitro", "in_silico", "ex_vivo", "not_specified"],
                                },
                                "dose": {"type": "string"},
                                "route": {
                                    "type": "string",
                                    "enum": ["oral", "ip", "iv", "sc", "topical", "not_specified"],
                                },
                                "duration": {"type": "string"},
                                "n_per_group": {"type": "string"},
                                "cell_line": {"type": "string"},
                                "glp": {
                                    "type": "boolean",
                                    "description": "True ONLY if text explicitly states GLP conditions",
                                },
                                "result_direction": {
                                    "type": "string",
                                    "enum": ["increased", "decreased", "no_change", "not_reached", "not_specified"],
                                },
                                "statistical_significance": {"type": "string"},
                            },
                        },
                        "source": {
                            "type": "object",
                            "required": ["section"],
                            "properties": {
                                "figure": {
                                    "type": "string",
                                    "description": "e.g. FIGURE 1, FIGURE 3A, FIGURE 6B — be specific",
                                },
                                "section": {
                                    "type": "string",
                                    "description": "abstract, introduction, methods, results, discussion, or figure_caption",
                                },
                                "direct_quote": {
                                    "type": "string",
                                    "description": "Verbatim quote from text supporting this, max 120 characters",
                                },
                            },
                        },
                    },
                },
            }
        },
        "required": ["relationships"],
    },
}

SYSTEM_PROMPT = """You are extracting information from a preclinical pharmacology paper about compound 11h, a novel PDE4 inhibitor.

Your task is to extract EVERY piece of information across these five categories:

1. PATHWAY BIOLOGY: Every directional relationship between biological entities. What activates, inhibits, produces, phosphorylates what. Every step in the mechanistic chain.

2. PHARMACOKINETICS: Every measurement of how the drug moves through the body. Cmax, AUC, t1/2, clearance, volume of distribution, brain/plasma ratios, tissue concentrations, accumulation on repeat dosing, bioavailability.

3. SAFETY AND TOLERABILITY: Every safety finding — adverse effects observed, adverse effects absent, cardiac safety, genotoxicity, cytotoxicity, behavioral effects, emesis data, weight changes, any tolerability observation.

4. STUDY DESIGN: The exact experimental conditions of every study — species, n per group, dose, route, duration, cell line, assay type, statistical method. These are templates for future experiment design.

5. MOLECULAR PROPERTIES: Structure, binding site, key molecular interactions, patent information, chemical properties, docking scores, ligand efficiency.

Rules:
- Extract from all five categories, not just pathway biology
- Every relationship needs a source — figure number or section
- Do NOT extract general biological knowledge not stated in this text
- Do NOT infer relationships not explicitly stated
- Mark GLP as true ONLY if the text explicitly says GLP
- For evidence_type: directly_measured means this section shows the experiment and data. paper_claims means it is asserted as a conclusion. cited_from_reference means it is attributed to another paper.
- Be exhaustive — it is better to extract too much than too little
- Quantitative values are especially important — capture every number"""


async def extract_section(
    section_name: str,
    text: str,
    figure_label: Optional[str] = None,
) -> list[dict]:
    """Single Claude API call — extracts all relationships from one section."""
    client = anthropic.AsyncAnthropic()
    user_content = f"Extract all information from this {'figure caption' if figure_label else section_name} section"
    if figure_label:
        user_content += f" ({figure_label})"
    user_content += f":\n\n{text}"

    try:
        response = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_information"},
            messages=[{"role": "user", "content": user_content}],
        )
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_block:
            return []
        return tool_block.input.get("relationships", [])
    except Exception as e:
        print(f"[stage2] extraction failed for {figure_label or section_name}: {e}")
        return []


async def run_extraction(paper: dict) -> list[dict]:
    """
    Fire all section + figure-caption extractions concurrently via asyncio.gather.
    Returns flat list of all relationship dicts with _extracted_from metadata added.
    """
    tasks = []
    meta = []  # parallel list tracking what each task is for

    sections = [
        ("abstract", paper.get("abstract") or ""),
        ("introduction", paper.get("introduction") or ""),
        ("methods", paper.get("methods_section") or ""),
        ("results", paper.get("results_section") or ""),
        ("discussion", paper.get("discussion_section") or ""),
    ]
    for section_name, text in sections:
        if text and len(text.strip()) > 100:
            tasks.append(extract_section(section_name, text))
            meta.append({"type": "section", "name": section_name, "label": None})

    for fig in paper.get("figure_captions", []):
        caption_text = (
            f"{fig.get('caption_title', '')} {fig.get('caption_text', '')}".strip()
        )
        if caption_text:
            label = fig.get("label", "")
            tasks.append(extract_section("figure_caption", caption_text, figure_label=label))
            meta.append({"type": "figure", "name": "figure_caption", "label": label})

    print(f"[stage2] launching {len(tasks)} concurrent extraction calls")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_relationships: list[dict] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[stage2] task {i} ({meta[i]['name']}) raised: {result}")
            continue
        m = meta[i]
        for r in result:
            r["_extracted_from"] = m["name"]
            if m["type"] == "figure" and m["label"]:
                src = r.setdefault("source", {})
                src["figure"] = m["label"]
                src["section"] = "figure_caption"
        all_relationships.extend(result)

    print(f"[stage2] total relationships extracted: {len(all_relationships)}")
    return all_relationships


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _canonical(raw: str) -> str:
    """Map a raw entity name to its canonical ID. Unknown → UNKNOWN_ prefix."""
    key = raw.lower().strip()
    cid = ENTITY_SYNONYMS.get(key)
    if cid:
        return cid
    # Try removing trailing 's' for plurals
    if key.endswith("s"):
        cid = ENTITY_SYNONYMS.get(key[:-1])
        if cid:
            return cid
    return "UNKNOWN_" + raw.upper().replace(" ", "_").replace("-", "_").replace("/", "_")


def _node_type(canonical_id: str) -> str:
    for prefix, t in NODE_TYPE_MAP.items():
        if canonical_id.startswith(prefix):
            return t
    return "other"


def build_graph(relationships: list[dict], paper: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph["pmid"] = paper.get("pmid")
    G.graph["title"] = paper.get("title")
    G.graph["year"] = paper.get("year")

    # Edge registry — deduplicates by (from, to, relationship verb)
    edge_registry: dict[tuple, dict] = {}
    evidence_priority = {"directly_measured": 3, "paper_claims": 2, "cited_from_reference": 1}

    for rel in relationships:
        raw_from = rel.get("from_entity", "").strip()
        raw_to = rel.get("to_entity", "").strip()
        if not raw_from or not raw_to:
            continue

        from_id = _canonical(raw_from)
        to_id = _canonical(raw_to)

        for node_id, raw_name in [(from_id, raw_from), (to_id, raw_to)]:
            if node_id not in G.nodes:
                G.add_node(
                    node_id,
                    canonical_id=node_id,
                    display_name=raw_name,
                    node_type=_node_type(node_id),
                    flagged=node_id.startswith("UNKNOWN_"),
                    category=rel.get("category", "unknown"),
                )

        relationship_verb = rel.get("relationship", "").strip()
        edge_key = (from_id, to_id, relationship_verb)
        conditions = rel.get("experimental_conditions") or {}
        source = rel.get("source") or {}

        if edge_key not in edge_registry:
            edge_registry[edge_key] = {
                "relationship": relationship_verb,
                "relationship_direction": rel.get("relationship_direction", "other"),
                "category": rel.get("category", "unknown"),
                "evidence_type": rel.get("evidence_type", "paper_claims"),
                "quantitative_values": [rel["quantitative_value"]] if rel.get("quantitative_value") else [],
                "experimental_conditions": [conditions] if conditions else [],
                "sources": [source] if source else [],
                "glp": bool(conditions.get("glp")),
            }
        else:
            existing = edge_registry[edge_key]
            if conditions:
                existing["experimental_conditions"].append(conditions)
            if source:
                existing["sources"].append(source)
            if rel.get("quantitative_value"):
                existing["quantitative_values"].append(rel["quantitative_value"])
            # Upgrade to strongest evidence type
            new_et = rel.get("evidence_type", "paper_claims")
            if evidence_priority.get(new_et, 0) > evidence_priority.get(existing["evidence_type"], 0):
                existing["evidence_type"] = new_et
            if conditions.get("glp"):
                existing["glp"] = True

    for (from_id, to_id, _), edge_data in edge_registry.items():
        G.add_edge(from_id, to_id, **edge_data)

    return G


def to_cytoscape(G: nx.DiGraph) -> dict:
    elements = []

    for node_id, data in G.nodes(data=True):
        elements.append({
            "group": "nodes",
            "data": {
                "id": node_id,
                "label": data.get("display_name", node_id),
                "type": data.get("node_type", "other"),
                "category": data.get("category", "unknown"),
                "flagged": data.get("flagged", False),
            },
        })

    for from_id, to_id, data in G.edges(data=True):
        edge_id = f"{from_id}__{to_id}__{data.get('relationship', '')}"
        elements.append({
            "group": "edges",
            "data": {
                "id": edge_id,
                "source": from_id,
                "target": to_id,
                "relationship": data.get("relationship", ""),
                "relationship_direction": data.get("relationship_direction", "other"),
                "category": data.get("category", "unknown"),
                "evidence_type": data.get("evidence_type", "paper_claims"),
                "glp": data.get("glp", False),
                "quantitative_values": data.get("quantitative_values", []),
                "experimental_conditions": data.get("experimental_conditions", []),
                "sources": data.get("sources", []),
            },
        })

    nodes_by_type = dict(Counter(
        data.get("node_type", "other") for _, data in G.nodes(data=True)
    ))
    edges_by_category = dict(Counter(
        data.get("category", "unknown") for _, _, data in G.edges(data=True)
    ))
    evidence_breakdown = dict(Counter(
        data.get("evidence_type", "paper_claims") for _, _, data in G.edges(data=True)
    ))
    glp_count = sum(1 for _, _, data in G.edges(data=True) if data.get("glp"))

    return {
        "elements": elements,
        "stats": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "nodes_by_type": nodes_by_type,
            "edges_by_category": edges_by_category,
            "evidence_breakdown": evidence_breakdown,
            "glp_edge_count": glp_count,
        },
    }


# ---------------------------------------------------------------------------
# Disk helpers — called from main.py
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def cache_path(pmid: str) -> Path:
    return DATA_DIR / "paper_cache" / f"{pmid}.json"


def graph_path(pmid: str) -> Path:
    return DATA_DIR / "graphs" / f"{pmid}.json"


async def build_graph_for_pmid(pmid: str) -> dict:
    """Top-level entry point: load cached paper, extract, build, save, return."""
    p = cache_path(pmid)
    if not p.exists():
        raise FileNotFoundError(f"No cached paper for PMID {pmid}. Run Stage 1 first.")

    with open(p) as f:
        paper = json.load(f)

    relationships = await run_extraction(paper)
    G = build_graph(relationships, paper)
    cyto = to_cytoscape(G)

    gp = graph_path(pmid)
    gp.parent.mkdir(parents=True, exist_ok=True)
    with open(gp, "w") as f:
        json.dump(cyto, f, indent=2)

    print(f"[stage2] graph saved → {gp}  ({cyto['stats']['node_count']} nodes, {cyto['stats']['edge_count']} edges)")
    return cyto
