import asyncio
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-6"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUTILS_PARAMS = {"tool": "INDAccelerator", "email": "ind@accelerator.com"}

# Drug class abbreviation → PubMed-indexed term
# PubMed uses full MeSH terms; abbreviations like "PDE4 inhibitor" return ~0 results
_DRUG_CLASS_EXPANSIONS = {
    "pde4 inhibitor":  "phosphodiesterase inhibitor",
    "pde5 inhibitor":  "phosphodiesterase inhibitor",
    "pde inhibitor":   "phosphodiesterase inhibitor",
    "jak inhibitor":   "janus kinase inhibitor",
    "btk inhibitor":   "bruton tyrosine kinase inhibitor",
    "mtor inhibitor":  "mTOR inhibitor",
    "pi3k inhibitor":  "phosphoinositide 3-kinase inhibitor",
    "cdk inhibitor":   "cyclin-dependent kinase inhibitor",
    "hdac inhibitor":  "histone deacetylase inhibitor",
    "ssri":            "serotonin reuptake inhibitor",
    "nsaid":           "anti-inflammatory agents non-steroidal",
}

def _pubmed_drug_class(drug_class: str) -> str:
    """Return a PubMed-friendly search term for a drug class."""
    key = drug_class.lower().strip()
    return _DRUG_CLASS_EXPANSIONS.get(key, drug_class)


# Map subsection names → targeted PubMed queries
# Use {drug_class} placeholder; it is expanded via _pubmed_drug_class() before searching
SUBSECTION_QUERIES = {
    "Primary Pharmacology":                           '{drug_class} pharmacology mechanism preclinical',
    "Secondary Pharmacology & Selectivity":           '{drug_class} selectivity off-target binding preclinical',
    "Safety Pharmacology — Cardiovascular (hERG/QT)": '{drug_class} hERG cardiovascular safety',
    "Safety Pharmacology — CNS":                      '{drug_class} CNS safety rat preclinical',
    "Safety Pharmacology — Respiratory":              '{drug_class} respiratory safety rat preclinical',
    "Pharmacokinetics & ADME":                        '{drug_class} pharmacokinetics ADME rat oral',
    "Single-Dose Toxicity":                           '{drug_class} acute toxicity rat oral',
    "Repeat-Dose Toxicity":                           '{drug_class} repeat dose toxicity rat',
    "Genotoxicity":                                   '{drug_class} genotoxicity Ames micronucleus',
    "Drug-Drug Interaction / CYP Profiling":          '{drug_class} CYP inhibition drug interaction',
}


# ── PubMed helpers ─────────────────────────────────────────────────────────────

async def _search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Return list of PMIDs for a PubMed query."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params={"db": "pubmed", "term": query,
                        "retmax": max_results, "retmode": "json",
                        "sort": "relevance", **EUTILS_PARAMS},
            )
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


async def _fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + truncated abstract for a list of PMIDs via XML."""
    if not pmids:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids),
                        "rettype": "abstract", "retmode": "xml", **EUTILS_PARAMS},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el    = article.find(".//PMID")
            title_el   = article.find(".//ArticleTitle")
            abs_els    = article.findall(".//AbstractText")
            year_el    = article.find(".//PubDate/Year")
            pmid   = pmid_el.text  if pmid_el   is not None else ""
            title  = title_el.text if title_el  is not None else ""
            year   = year_el.text  if year_el   is not None else ""
            abstract = " ".join((el.text or "") for el in abs_els if el.text)[:600]
            if pmid and title:
                results.append({"pmid": pmid, "title": title,
                                 "year": year, "abstract": abstract})
        return results
    except Exception:
        return []


async def _fetch_precedents(drug_class: str, subsection: str) -> dict:
    """Search PubMed for precedent studies for one subsection."""
    expanded = _pubmed_drug_class(drug_class)
    template = SUBSECTION_QUERIES.get(subsection, '{drug_class} preclinical study')
    query = template.format(drug_class=expanded)
    pmids = await _search_pubmed(query, max_results=5)
    papers = await _fetch_abstracts(pmids)
    return {"subsection": subsection, "search_query": query, "papers": papers}


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a GLP study design expert and CRO liaison with 20 years of preclinical drug development experience.
Output ONLY newline-delimited JSON objects (NDJSON). Each brief as a separate top-level JSON object.
No wrapper arrays, no markdown, no commentary."""

BRIEF_PROMPT = """Generate a concise study design brief for each study below.
These briefs will be sent to CROs to get quotes and initiate GLP protocol development.

COMPOUND: COMPOUND_PLACEHOLDER
DRUG CLASS: DRUG_CLASS_PLACEHOLDER
TARGET: TARGET_PLACEHOLDER
INDICATION: INDICATION_PLACEHOLDER

COMPOUND PROFILE (for comparability assessment):
PROFILE_PLACEHOLDER

PAPER EXPERIMENTS (what has already been done):
PAPER_EXPERIMENTS_PLACEHOLDER

PUBLISHED PRECEDENTS FROM PUBMED (fetched live — one block per study subsection):
PRECEDENTS_PLACEHOLDER

TIMELINE STUDIES (what needs to be commissioned):
STUDIES_PLACEHOLDER

For each study in the timeline, output ONE JSON object with this exact schema:

{"type":"brief","study_id":"<S01>","study_name":"<name>","regulatory_basis":"<exact ICH/CFR clause>","objective":"<one sentence>","species":"<e.g. Sprague-Dawley rat>","strain_sex":"<e.g. Male and female, n=10/sex>","route":"<e.g. Oral gavage>","dose_levels":"<e.g. 0 (vehicle), 10, 30, 100 mg/kg>","dose_rationale":"<1-2 sentences citing paper data AND precedent studies>","duration":"<e.g. 28 days + 14-day recovery>","primary_endpoints":["<endpoint 1>"],"secondary_endpoints":["<endpoint 1>"],"glp_required":<bool>,"acceptance_criteria":"<1 sentence: what a passing result looks like>","cro_minimum_requirements":["<requirement 1>"],"data_deliverables":["<deliverable 1>"],"paper_connection":"<1 sentence citing specific EXP IDs>","fail_consequence":"<1 sentence>","cro_notes":"<1 sentence>","pubmed_search_query":"<the search query used for this subsection>","precedents":[{"pmid":"<12345678>","title":"<title>","year":"<2019>","comparable":true,"comparability_note":"<1 sentence: why comparable or not — cite specific factors: potency, route, species, duration>","doses_extracted":"<e.g. 3/10/30 mg/kg oral rat 28-day>","noael_extracted":"<e.g. 10 mg/kg or unknown>"}]}

COMPARABILITY ASSESSMENT RULES:
- For each precedent paper, assess comparability to THIS compound based on:
  1. Drug class match (same mechanism = high relevance)
  2. Potency match (within 10× IC50 = comparable; >100× different = flag)
  3. Route match (oral vs IP vs IV — must match proposed clinical route)
  4. Species match (rat vs mouse vs dog)
  5. CNS penetration (brain-penetrant vs peripherally-acting compounds have different dose-limiting toxicities)
- Set comparable: true only if at least drug class + route + species match
- Always explain WHY in comparability_note — never just say "comparable" without specifics
- If a precedent used a much lower potency compound: note "doses will need adjustment — [precedent compound] IC50 ~Xμm vs this compound IC50 ~XnM"
- Include ALL returned papers in the precedents array (comparable: false ones too — they show the scientist what was considered and excluded)
- Use comparable: true precedents to JUSTIFY dose levels in dose_rationale
- If no comparable precedents found: set precedents: [] and derive doses from paper data alone

RULES:
- Use paper experiment data to justify dose levels wherever possible
- GLP required = true for: safety pharmacology core battery, genotoxicity, repeat-dose tox
- GLP required = false for: dose-ranging pilots, exploratory PK, non-pivotal screening assays
- Species: use rat for most studies unless specific reason requires dog/minipig
- Route: always match the proposed clinical route
- acceptance_criteria: frame in terms of multiples of expected clinical dose
- cro_notes: note if this study can be combined with another (e.g. CNS FOB + respiratory same day per ICH S7A §4.4)

INDICATION-SPECIFIC DESIGN RULES:
- If indication is Alzheimer's or CNS neurodegeneration:
  * CNS distribution study: endpoints must include brain tissue, CSF, and plasma LC-MS/MS at multiple timepoints; acceptance_criteria = "brain:plasma ratio >= 0.3 at Cmax required to proceed"
  * CNS FOB study: primary_endpoints must include novel object recognition (NOR) index and Y-maze spontaneous alternation %
  * CYP DDI brief: explicitly flag CYP2D6 (donepezil is a CYP2D6 substrate)
- If drug class is PDE4 inhibitor:
  * All repeat-dose tox briefs: primary_endpoints must include GI histopathology (stomach, duodenum, jejunum), body weight, food consumption
  * dose_rationale must note: "GI tolerability expected to limit maximum tolerated dose; dose titration recommended"
  * fail_consequence for repeat-dose tox: "GI pathology at therapeutic multiples may require dose titration or formulation change before Phase 1"

Output one JSON brief per study. Order by study_id (S01, S02...).
"""


# ── Main streaming function ────────────────────────────────────────────────────

async def stream_study_briefs(studies: list[dict], paper_experiments: list[dict],
                               compound: str, drug_class: str, target: str,
                               indication: str, profile: dict | None = None):
    """Stream one study design brief per study, grounded in PubMed precedents."""

    # 1. Identify unique subsections and fetch PubMed precedents in parallel
    subsections = list({s.get("subsection", "") for s in studies if s.get("subsection")})
    precedent_results = await asyncio.gather(
        *[_fetch_precedents(drug_class, sub) for sub in subsections],
        return_exceptions=True,
    )
    # Build subsection → precedent map
    precedents_by_sub: dict[str, dict] = {}
    for result in precedent_results:
        if isinstance(result, dict):
            precedents_by_sub[result["subsection"]] = result

    # 2. Format precedents block for the prompt
    precedents_text_parts = []
    for sub, data in precedents_by_sub.items():
        papers = data.get("papers", [])
        if not papers:
            continue
        block = f"Subsection: {sub}\nSearch query: {data['search_query']}\n"
        for p in papers:
            block += (f"  PMID {p['pmid']} ({p['year']}): {p['title']}\n"
                      f"  Abstract: {p['abstract']}\n\n")
        precedents_text_parts.append(block)
    precedents_text = "\n---\n".join(precedents_text_parts) if precedents_text_parts \
        else "[No PubMed results retrieved — use compound profile and paper data for dose justification]"

    # 3. Build profile summary for comparability context
    profile_text = json.dumps(profile, indent=2) if profile else \
        f"compound={compound}, drug_class={drug_class}, target={target}, indication={indication}"

    # 4. Build prompt using .replace() — avoids KeyError from JSON {} in context
    prompt = (BRIEF_PROMPT
              .replace("COMPOUND_PLACEHOLDER",          compound)
              .replace("DRUG_CLASS_PLACEHOLDER",        drug_class)
              .replace("TARGET_PLACEHOLDER",            target)
              .replace("INDICATION_PLACEHOLDER",        indication)
              .replace("PROFILE_PLACEHOLDER",           profile_text)
              .replace("PAPER_EXPERIMENTS_PLACEHOLDER", json.dumps(paper_experiments, indent=2))
              .replace("PRECEDENTS_PLACEHOLDER",        precedents_text)
              .replace("STUDIES_PLACEHOLDER",           json.dumps(studies, indent=2)))

    # 5. Stream briefs from Claude
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    depth = 0
    obj = ""
    in_obj = False
    all_briefs = []

    async with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for chunk in stream.text_stream:
            for char in chunk:
                if char == '{':
                    if depth == 0:
                        obj = '{'
                        in_obj = True
                    elif in_obj:
                        obj += char
                    depth += 1
                elif char == '}' and in_obj:
                    depth -= 1
                    obj += char
                    if depth == 0:
                        try:
                            parsed = json.loads(obj)
                            if parsed.get("type") == "brief":
                                all_briefs.append(parsed)
                                yield parsed
                        except json.JSONDecodeError:
                            pass
                        obj = ""
                        in_obj = False
                elif in_obj:
                    obj += char

    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_DIR / "study_briefs.json", "w") as f:
        json.dump({"briefs": all_briefs}, f, indent=2)
