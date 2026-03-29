import json
import os
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()


async def _fetch_ct_precedents(drug_class: str, indication: str) -> str:
    """Fetch 2-3 Phase 1 precedent trials from ClinicalTrials.gov for protocol grounding."""
    try:
        query = f"{drug_class} {indication}".strip()
        params = {
            "query.term": query,
            "filter.phase": "PHASE1",
            "filter.studyType": "INTERVENTIONAL",
            "fields": "protocolSection.identificationModule,protocolSection.designModule,protocolSection.eligibilityModule,protocolSection.armsInterventionsModule",
            "pageSize": 3,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://clinicaltrials.gov/api/v2/studies", params=params)
            resp.raise_for_status()
            data = resp.json()
        studies = data.get("studies", [])
        if not studies:
            return ""
        parts = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            design = proto.get("designModule", {})
            eligibility = proto.get("eligibilityModule", {})
            nct = ident.get("nctId", "")
            title = ident.get("briefTitle", "")
            phases = design.get("phases", [])
            enrollment = design.get("enrollmentInfo", {}).get("count", "")
            criteria = eligibility.get("eligibilityCriteria", "")[:400]
            parts.append(f"NCT: {nct}\nTitle: {title}\nPhases: {phases}\nEnrollment: {enrollment}\nKey eligibility: {criteria}...")
        return "CLINICALTRIALS.GOV PRECEDENTS:\n" + "\n---\n".join(parts)
    except Exception:
        return ""  # graceful fallback — draft proceeds without CT data

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a regulatory affairs writer with 15 years of IND filing experience at FDA-regulated biotech and pharma companies.
You write clear, precise regulatory documents that follow FDA conventions and 21 CFR §312.23 structure.
Write in active voice. Use proper regulatory language. Do not hedge excessively.
Output plain text with markdown headers (## for section, ### for subsection). No JSON."""

SECTION_META = {
    1:  {"title": "Cover Sheet (FDA Form 1571)", "ref": "21 CFR §312.23(a)(1)", "can_draft": True},
    2:  {"title": "Table of Contents", "ref": "21 CFR §312.23(a)(2)", "can_draft": True},
    3:  {"title": "Introductory Statement and General Investigational Plan", "ref": "21 CFR §312.23(a)(3)", "can_draft": True},
    4:  {"title": "General Investigational Plan", "ref": "21 CFR §312.23(a)(4)", "can_draft": True},
    5:  {"title": "Investigator's Brochure", "ref": "21 CFR §312.23(a)(5)", "can_draft": True},
    6:  {"title": "Clinical Protocols — Phase 1 SAD/MAD", "ref": "21 CFR §312.23(a)(6)", "can_draft": True},
    7:  {"title": "Chemistry, Manufacturing and Controls (CMC)", "ref": "21 CFR §312.23(a)(7)", "can_draft": True},
    8:  {"title": "Pharmacology and Toxicology Data Summary", "ref": "21 CFR §312.23(a)(8)", "can_draft": True},
    9:  {"title": "Previous Human Experience", "ref": "21 CFR §312.23(a)(9)", "can_draft": True},
    10: {"title": "Additional Information", "ref": "21 CFR §312.23(a)(10)", "can_draft": True},
}

DRAFT_PROMPTS = {
    1: """Draft a completed FDA Form 1571 cover sheet template for an IND application.

COMPOUND DATA:
{context}

Instructions:
- Format as a structured template with field names and pre-filled values where data is available
- Mark fields requiring sponsor-specific data as [SPONSOR TO COMPLETE]
- Mark fields requiring IND number as [ASSIGNED BY FDA]
- Include: Sponsor name/address field, compound name, proposed indication, phase of study, submission type (initial IND), date field
- Note at top: "NOTE: This is a pre-populated draft. Fields marked [SPONSOR TO COMPLETE] require sponsor information before submission."
- Follow FDA Form 1571 field order exactly""",

    2: """Draft a complete Table of Contents for an IND application.

COMPOUND DATA:
{context}

Instructions:
- Follow 21 CFR §312.23(a) section order (sections 1-10)
- For each section, list the section number, title, and subsections
- Mark section status: [COMPLETE], [PARTIAL — requires additional data], or [PENDING — study data required]
- For Section 8, list all 10 pharmacology/toxicology subsections
- Include page number placeholders as [p. XX]
- Professional regulatory document format""",

    3: """Draft Section 3: Introductory Statement and General Investigational Plan.

COMPOUND DATA:
{context}

Instructions:
Draft these subsections:
## 3.1 Introductory Statement
- Name and description of the compound (structure class, mechanism, target)
- Prior investigations — what has been done and by whom (cite paper data)
- General investigational plan overview

## 3.2 General Investigational Plan
- Phase 1 objectives (safety, tolerability, PK characterization)
- Study population description (healthy volunteers vs patients — justify choice)
- Estimated number of subjects
- Duration of individual subject participation
- Anticipated duration to complete Phase 1

## 3.3 Scientific Rationale
- Disease/condition rationale
- Compound mechanism and why it addresses the disease
- Translational rationale connecting preclinical findings to expected human effects

Write as a complete regulatory document section. Be specific — use compound name, mechanism, and indication from the data provided. Cite specific experimental findings where they support claims.""",

    4: """Draft Section 4: General Investigational Plan.

COMPOUND DATA:
{context}

Instructions:
Draft a comprehensive investigational plan covering:

## 4.1 Phase 1 Program Objectives
- Primary: safety and tolerability in humans
- Secondary: pharmacokinetic characterization
- Exploratory: pharmacodynamic markers if applicable

## 4.2 Phase 1 Study Design Overview
- SAD (Single Ascending Dose) cohort structure: proposed number of cohorts, dose levels, dose increments
- MAD (Multiple Ascending Dose) cohort structure
- Starting dose justification (NOAEL-based using preclinical data, apply species scaling)
- Maximum proposed dose justification

## 4.3 Subject Population
- Inclusion criteria (age, health status, relevant biomarkers)
- Exclusion criteria (comorbidities, medications, safety exclusions specific to this drug class)

## 4.4 Safety Monitoring Plan
- Stopping rules (individual subject and cohort-level)
- Safety review committee structure
- Sentinel dosing approach (2 subjects before full cohort)

## 4.5 Proposed Phase 2 Program (brief)
- Intended patient population for Phase 2
- Key Phase 1 data needed before Phase 2 initiation

Be specific. Derive starting dose from the preclinical NOAEL/MTD data provided. Apply a conservative human equivalent dose conversion (HED) using standard body surface area scaling (rat-to-human factor of 6.2).""",

    5: """Draft an Investigator's Brochure (IB) skeleton for Phase 1.

COMPOUND DATA:
{context}

Instructions:
Draft the IB structure and populate sections where data exists:

## 1. Summary
- One-page executive summary of compound, mechanism, preclinical findings, and proposed Phase 1 plan

## 2. Introduction
- Compound identity (name, structure class, molecular formula if available)
- Mechanism of action
- Target indication and unmet medical need

## 3. Physical, Chemical and Pharmaceutical Properties

## 4. Nonclinical Studies
### 4.1 Pharmacology
- Primary pharmacology: summarize in vitro and in vivo PD findings from paper
- Secondary pharmacology: selectivity data
### 4.2 Pharmacokinetics and Drug Metabolism
- Summarize available PK data
- Note gaps requiring further characterization
### 4.3 Toxicology
- Summarize completed toxicology studies
- Flag pending studies in table format: Study | Status | Estimated Completion

## 5. Effects in Humans
- [Section to be completed as clinical data accumulates]

## 6. Summary of Data and Guidance for the Investigator
- Key safety signals to monitor
- Suggested precautions

Populate each section with actual data from the compound context provided. For missing studies, write "[Data pending — study commissioned: Expected Q[X] 20XX]".""",

    6: """Draft a Phase 1 clinical protocol skeleton (SAD/MAD dose escalation).

COMPOUND DATA:
{context}

Instructions:
Draft a Phase 1 first-in-human protocol outline. This is a skeleton — clinical operations will finalize.

## Protocol Title
## Synopsis Table
(Protocol number, phase, objectives, design, population, dose range, endpoints, duration)

## 1. Background and Rationale
- Disease background (2-3 sentences)
- Compound mechanism and preclinical rationale
- Justification for first-in-human study

## 2. Study Objectives
### 2.1 Primary Objectives
- Safety and tolerability
### 2.2 Secondary Objectives
- PK parameters (Cmax, AUC, t1/2, Tmax)
### 2.3 Exploratory Objectives
- PD biomarkers if applicable

## 3. Study Design
### 3.1 SAD Phase
- Number of cohorts: [derive from dose range]
- Dose levels: [derive from preclinical NOAEL; starting dose = NOAEL × HED correction ÷ 10 safety factor]
- Subjects per cohort: 8 (6 active + 2 placebo), single-blind
- Sentinel dosing: 2 subjects (1 active + 1 placebo) 48h before remaining cohort
- Dose escalation criteria

### 3.2 MAD Phase
- Number of cohorts, duration of dosing, dose levels based on SAD results

## 4. Subject Selection
### 4.1 Inclusion Criteria
### 4.2 Exclusion Criteria (include class-specific exclusions)

## 5. Dose Escalation Rules and Stopping Criteria
- Individual stopping criteria (specific AEs)
- Cohort stopping criteria (DLT definitions)
- Program-stopping criteria

## 6. Safety Assessments
- Schedule of assessments table
- Key safety parameters (vitals, labs, ECG, physical exam)
- Any indication-specific monitoring

## 7. Pharmacokinetic Assessments
- Sampling schedule for SAD and MAD
- Key PK parameters to be determined

## 8. Statistical Considerations
- Sample size justification
- Analysis populations

## Appendix A: Dose Justification
- Show NOAEL → HED calculation with numbers from preclinical data

Use SPECIFIC numbers from the compound data. Derive the starting dose mathematically. Show the calculation.""",

    7: """Draft Section 7: Chemistry, Manufacturing and Controls (CMC).

COMPOUND DATA:
{context}

Instructions:
Draft what can be completed from available data, flag what requires sponsor input:

## 7.1 Drug Substance
### 7.1.1 Description and Characterization
- Chemical name (IUPAC and common)
- Molecular formula and weight
- Physical/chemical properties
- Structural description

### 7.1.2 Manufacturer
[SPONSOR TO COMPLETE: GMP manufacturer name and address]

### 7.1.3 Synthesis
- General synthetic route description (if in paper)
- Key intermediates

### 7.1.4 Analytical Controls
- [List analytical methods that should be established]

### 7.1.5 Stability
- [Framework for stability program]

## 7.2 Drug Product (Formulation)
### 7.2.1 Composition
- Proposed dosage form
- Excipients

### 7.2.2 Manufacture
[SPONSOR TO COMPLETE]

### 7.2.3 Analytical Controls

## 7.3 Placebo (if applicable)

Note clearly which fields are pre-populated from paper data vs. [SPONSOR TO COMPLETE].""",

    8: """Draft Section 8: Pharmacology and Toxicology — Integrated Summary.

COMPOUND DATA:
{context}

Instructions:
Draft the integrated nonclinical summary for IND submission:

## 8.1 Pharmacology
### 8.1.1 Primary Pharmacodynamics
Summarize target engagement, mechanism, dose-response. Cite specific study findings.

### 8.1.2 Secondary Pharmacodynamics and Selectivity
Summarize off-target data. Flag any selectivity concerns.

### 8.1.3 Safety Pharmacology
Summarize completed safety pharm studies. For each subsection:
- Cardiovascular: [data / pending]
- CNS: [data / pending]
- Respiratory: [data / pending]

## 8.2 Pharmacokinetics and ADME
Summarize PK data by species and route. Include key parameters table.

## 8.3 Toxicology Studies
### 8.3.1 Single-Dose Toxicity
### 8.3.2 Repeat-Dose Toxicity
### 8.3.3 Genotoxicity
### 8.3.4 Other Toxicity Studies

## 8.4 Integrated Safety Assessment
- NOAEL identification and species comparison
- Safety margins at proposed clinical starting dose
- Key monitoring parameters for Phase 1

## 8.5 Pending Studies Table
| Study | Regulatory Basis | Status | Expected Completion |
|-------|-----------------|--------|---------------------|
For each missing study from the IND map gaps.

Use the actual experimental data provided. For completed studies cite findings; for pending studies write "[Pending — required before IND submission]" or "[Pending — Phase 2 requirement]".""",

    9: """Draft Section 9: Previous Human Experience.

COMPOUND DATA:
{context}

Instructions:
## 9.1 Prior Human Studies
[If the compound class has marketed drugs (e.g., other PDE4 inhibitors), note the class-level human experience and its relevance]

## 9.2 Published Clinical Literature
[Summarize any published human data on this compound or closely related analogs]

## 9.3 Marketed Drugs in Same Class
[Summarize approved drugs in the same drug class, their clinical safety profile, and how this compound compares or differentiates]

## 9.4 Adverse Events of Interest
[Based on class effects, flag known human AEs from approved drugs in the class]

If no direct human data: clearly state "No prior human experience with this specific compound. Class-level human data summarized below." then provide the class data.""",

    10: """Draft Section 10: Additional Information.

COMPOUND DATA:
{context}

Instructions:
Assess and draft any applicable subsections:

## 10.1 Controlled Substance Scheduling
- Assess: does this compound have abuse potential? [PDE4 inhibitors typically do not]
- If no: "This compound does not appear to have abuse potential based on its mechanism..."

## 10.2 Radioactive Drug
- If applicable. If not: brief statement that no radioactive dosing is planned for Phase 1.

## 10.3 Pediatric Studies
- State: "Pediatric studies are not planned for Phase 1. A pediatric investigation plan will be submitted per applicable regulations prior to Phase 2."

## 10.4 Expedited Development Programs
- Assess eligibility for Fast Track, Breakthrough Therapy, Accelerated Approval based on indication
- Note: this requires formal FDA interaction to confirm

## 10.5 Other Relevant Information
- Any other information relevant to this specific compound or development program"""
}


async def stream_ind_draft(section: int):
    """Stream an IND section draft. Reads pipeline state to construct context."""
    # Load all available pipeline state
    context_parts = []

    profile_path = PIPELINE_STATE_DIR / "profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
        context_parts.append(f"COMPOUND PROFILE:\n{json.dumps(profile, indent=2)}")

    paper_path = PIPELINE_STATE_DIR / "paper_data.json"
    if paper_path.exists():
        with open(paper_path) as f:
            paper = json.load(f)
        context_parts.append(f"PAPER EXPERIMENTS:\n{json.dumps(paper.get('experiments', []), indent=2)}")

    ind_path = PIPELINE_STATE_DIR / "ind_map.json"
    if ind_path.exists():
        with open(ind_path) as f:
            ind_map = json.load(f)
        context_parts.append(f"IND MAP (gap assessment):\n{json.dumps(ind_map, indent=2)}")

    timeline_path = PIPELINE_STATE_DIR / "timeline.json"
    if timeline_path.exists():
        with open(timeline_path) as f:
            timeline = json.load(f)
        context_parts.append(f"PRECLINICAL TIMELINE:\n{json.dumps(timeline, indent=2)}")

    briefs_path = PIPELINE_STATE_DIR / "study_briefs.json"
    if briefs_path.exists():
        with open(briefs_path) as f:
            briefs = json.load(f)
        context_parts.append(f"STUDY DESIGN BRIEFS:\n{json.dumps(briefs, indent=2)}")

    # For Section 6 (clinical protocol), enrich context with ClinicalTrials.gov precedents
    if section == 6:
        drug_class = profile.get("drug_class", "") if profile_path.exists() else ""
        indication = profile.get("indication", "") if profile_path.exists() else ""
        ct_data = await _fetch_ct_precedents(drug_class, indication)
        if ct_data:
            context_parts.append(ct_data)

    context = "\n\n---\n\n".join(context_parts)

    meta = SECTION_META.get(section, {"title": f"Section {section}", "ref": ""})
    draft_prompt_template = DRAFT_PROMPTS.get(section)
    if not draft_prompt_template:
        yield f"Draft not available for Section {section}."
        return

    # Use replace() instead of .format() — context contains JSON with {/} chars
    user_prompt = draft_prompt_template.replace("{context}", context)
    full_prompt = f"""Draft IND Section {section}: {meta['title']}
Regulatory reference: {meta['ref']}

{user_prompt}"""

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_prompt}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            yield text_chunk
