import json
import os
import re
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-6"
ECFR_URL = "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312"
ICH_URL  = "https://database.ich.org/sites/default/files/M3_R2__Guideline.pdf"


async def _fetch_ecfr() -> str:
    """Fetch 21 CFR Part 312 text from eCFR."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(
                ECFR_URL,
                headers={"User-Agent": "INDAccelerator/1.0 (regulatory research tool)"},
            )
            resp.raise_for_status()
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text)
            # Grab just the §312.23 section
            idx = text.find("312.23")
            return text[max(0, idx - 200): idx + 12000] if idx != -1 else text[:12000]
    except Exception:
        return ""  # fall back to Claude's built-in knowledge


SYSTEM_PROMPT = """You are an FDA regulatory expert building an IND readiness assessment.
You have deep knowledge of 21 CFR Part 312, ICH M3(R2), ICH S7A/S7B, ICH S2(R1), and all relevant nonclinical guidance.
Output ONLY newline-delimited JSON objects (NDJSON) — no wrapper array, no markdown, no commentary.
Each section is a separate top-level JSON object on its own line."""

MAP_PROMPT = """Build a complete IND filing map for 21 CFR §312.23(a) all 10 sections.

REGULATORY REFERENCE (live eCFR text):
{ecfr_text}

EXTRACTED EXPERIMENTS FROM PAPER:
{experiments}

Output each section as a SEPARATE standalone JSON object. Do NOT wrap them in any outer object or array.
Output one JSON object per section, followed by a final summary object.

For sections 1-7 and 9-10: apply the STATUS RULES below to assign complete/partial/missing — NEVER use "not_started".
For section 8: map every experiment to its exact regulatory subsection (full detail below).

Section schema (output one at a time):
{{"number": <int>, "title": <str>, "regulatory_ref": <str>, "status": <str>, "description": <str>, "documents_needed": [{{"name": <str>, "status": "complete"|"partial"|"missing", "notes": <str>}}], "subsections": [...], "experiments": []}}

Output sections 1-10 in order, then:
{{"summary": "<2-3 sentence overall IND readiness assessment>"}}

STATUS RULES for sections 1-7 and 9-10:
- "complete": ALL required content can be written or generated right now from available information — including sections that are purely structural/formulaic
- "partial": some content is available or generatable, but gaps remain that require external information
- "missing": requires information that is fundamentally unavailable from the paper (sponsor-specific data, clinical protocols, manufacturing records, signatures)

Apply these section-by-section:
- Section 1 (Cover Sheet): "partial" — compound name, drug class, indication, proposed phase can be pre-populated from the paper; missing: sponsor name/address, IND number, authorized signature
- Section 2 (Table of Contents): "complete" — this is a structural document that can be fully generated right now from the standard IND section list; no external data required
- Section 3 (Introductory Statement & Plan): "complete" if paper clearly identifies compound, drug class, mechanism of action, and proposed indication (this narrative can be written from paper content); "partial" if only some of these are present; "missing" if paper lacks compound identity
- Section 4 (General Investigational Plan): "complete" if paper provides indication, proposed phase, study rationale, and estimated number/duration of studies (a plan document can be drafted from this); "partial" if rationale is present but phase/duration are unclear; "missing" if paper is purely mechanistic with no translational context
- Section 5 (Investigator's Brochure): "partial" — the paper's experimental data forms the scientific backbone of an IB, but a full IB requires additional compiled clinical/preclinical summary sections not in a single paper
- Section 6 (Protocols): "missing" — clinical trial protocols require IRB approval, investigator details, statistical analysis plans, and operational details that cannot come from a research paper
- Section 7 (Chemistry, Manufacturing & Controls): "partial" if paper describes synthesis route, formulation, purity/analytical data, or stability; "missing" if no CMC-relevant data present
- Section 9 (Previous Human Experience): "complete" or "partial" if paper references human clinical data or prior trials; "missing" if purely preclinical with no human data referenced
- Section 10 (Additional Information): "complete" if no special categories apply to this compound (most small molecules); "partial" if paper discusses relevant special considerations (e.g., abuse potential, radioactive labeling) that need formal documentation

For each section include documents_needed listing what is still required for IND submission.

Example (fill in actual assessed values — NEVER use "not_started", always use complete/partial/missing):
{{"number": 1, "title": "Cover Sheet", "regulatory_ref": "21 CFR §312.23(a)(1)", "status": "partial", "description": "FDA Form 1571 signed by sponsor.", "documents_needed": [{{"name": "Form FDA 1571 (compound/indication fields)", "status": "partial", "notes": "Compound name and indication can be pre-populated from paper; sponsor address and IND number unknown"}}, {{"name": "Authorized sponsor signature", "status": "missing", "notes": "Requires actual sponsor identity"}}, {{"name": "Proposed clinical indication", "status": "complete", "notes": "Derived from paper"}}], "subsections": [], "experiments": []}}

Section 8 full schema — fill in all subsections:
{{
  "number": 8,
  "title": "Pharmacology & Toxicology — Study Data",
  "regulatory_ref": "21 CFR §312.23(a)(8) · ICH M3(R2)",
  "status": "COMPUTE_FROM_SUBSECTIONS",
  "description": "All individual nonclinical pharmacology and toxicology studies.",
  "documents_needed": [],
  "experiments": [],
  "subsections": [
    {{"name": "Primary Pharmacology", "regulatory_ref": "21 CFR §312.23(a)(8)(i) · ICH M3(R2) §4", "description": "Mechanism of action, target engagement, dose-response in relevant in vitro and in vivo models.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Secondary Pharmacology & Selectivity", "regulatory_ref": "ICH M3(R2) §4 · ICH S7A §4.3", "description": "Off-target binding panel (>50 targets recommended), secondary pharmacodynamic effects.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Safety Pharmacology — Cardiovascular (hERG/QT)", "regulatory_ref": "ICH S7A §4.4 · ICH S7B", "description": "hERG channel inhibition, cardiac action potential, in vivo cardiovascular telemetry.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Safety Pharmacology — CNS", "regulatory_ref": "ICH S7A §4.4", "description": "FOB or modified Irwin test: motor activity, coordination, sensory/motor reflexes, autonomic function.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Safety Pharmacology — Respiratory", "regulatory_ref": "ICH S7A §4.4", "description": "Respiratory rate and tidal volume (or O2 saturation). Typically plethysmography in rat.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Pharmacokinetics & ADME", "regulatory_ref": "21 CFR §312.23(a)(8)(ii) · ICH M3(R2) §9", "description": "Single and repeat-dose PK, tissue distribution, plasma protein binding, in vitro metabolism.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Single-Dose Toxicity", "regulatory_ref": "ICH M3(R2) §6 · ICH S4", "description": "MTD or maximum feasible dose in one rodent species, same route as proposed clinical.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Repeat-Dose Toxicity", "regulatory_ref": "ICH M3(R2) §6 · ICH S4", "description": "Minimum 2-week GLP repeat-dose in one rodent species. Duration ≥ 2× clinical trial duration.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Genotoxicity", "regulatory_ref": "ICH S2(R1) · ICH M3(R2) §7", "description": "Ames test + in vitro chromosomal aberration or micronucleus. In vivo micronucleus if in vitro positive.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}},
    {{"name": "Drug-Drug Interaction / CYP Profiling", "regulatory_ref": "FDA DDI Guidance (2020) · ICH M3(R2) §9", "description": "In vitro CYP inhibition and induction for CYP1A2, 2C9, 2C19, 2D6, 3A4.", "status": "COMPUTE", "status_rationale": "", "experiments": [], "missing_experiments": []}}
  ]
}}

Rules for Section 8 subsections:
- Map each experiment ID (from extracted experiments) to the correct subsection(s) based on what it measured
- status: "complete" = sufficient for Phase 1 IND submission per ICH M3(R2) — does NOT mean every possible experiment is done, only that the data package justifies human exposure; "partial" = some data exists but a Phase 1-blocking gap remains; "missing" = no relevant study exists
- Write specific status_rationale (1-2 sentences citing the gap or why it's complete)
- missing_experiments: only list studies that are actually Phase 1 blocking or conditionally required; note "Phase 2" or "recommended" for non-blocking enhancements
- Non-GLP alone is NOT sufficient for genotoxicity or safety pharmacology core battery — these require GLP
- 28-day rat non-GLP repeat-dose = partial (GLP replication required)
- Ames alone (especially weak positive) = partial
- No respiratory study = missing
- Primary pharmacology: if robust in vitro target engagement (IC50/binding) plus at least one in vivo pharmacodynamic model are present, status = "complete" for Phase 1 — additional cellular assays or second in vivo models are recommended but not Phase 1 blocking
- PK/ADME: single-dose PK in one species is sufficient for Phase 1; full ADME, tissue distribution, and DDI panels can be deferred to Phase 2"""


async def stream_ind_map(experiments: list[dict]):
    """Stream one section JSON object at a time."""
    ecfr_text = await _fetch_ecfr()
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = MAP_PROMPT.format(
        ecfr_text=ecfr_text[:6000] if ecfr_text else "[Use built-in knowledge of 21 CFR §312.23 and ICH M3(R2)]",
        experiments=json.dumps(experiments, indent=2),
    )

    depth = 0
    obj = ""
    in_obj = False
    all_sections = []
    summary_text = ""

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
                            if "number" in parsed:
                                # Compute sec 8 status from subsections
                                if parsed["number"] == 8:
                                    statuses = [s.get("status") for s in parsed.get("subsections", [])]
                                    if statuses:
                                        if all(s == "complete" for s in statuses):
                                            parsed["status"] = "complete"
                                        elif all(s == "missing" for s in statuses):
                                            parsed["status"] = "missing"
                                        else:
                                            parsed["status"] = "partial"
                                all_sections.append(parsed)
                                yield parsed
                            elif "summary" in parsed:
                                summary_text = parsed["summary"]
                        except json.JSONDecodeError:
                            pass
                        obj = ""
                        in_obj = False
                elif in_obj:
                    obj += char

    # Save full result
    result = {"sections": all_sections, "summary": summary_text}
    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_DIR / "ind_map.json", "w") as f:
        json.dump(result, f, indent=2)


async def build_ind_map(experiments: list[dict]) -> dict:
    ecfr_text = await _fetch_ecfr()

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = MAP_PROMPT.format(
        ecfr_text=ecfr_text[:6000] if ecfr_text else "[Use built-in knowledge of 21 CFR §312.23 and ICH M3(R2)]",
        experiments=json.dumps(experiments, indent=2),
    )

    msg = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    result = json.loads(raw)

    # Compute Section 8 overall status from subsections
    for sec in result.get("sections", []):
        if sec["number"] == 8:
            statuses = [s["status"] for s in sec.get("subsections", [])]
            if all(s == "complete" for s in statuses):
                sec["status"] = "complete"
            elif all(s == "missing" for s in statuses):
                sec["status"] = "missing"
            else:
                sec["status"] = "partial"
            break

    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_DIR / "ind_map.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
