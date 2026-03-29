import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an FDA regulatory expert and preclinical development strategist.
Output ONLY newline-delimited JSON objects (NDJSON). Each study as a separate top-level JSON object.
No wrapper arrays, no markdown, no commentary."""

TIMELINE_PROMPT = """Create a Phase 1 preclinical development timeline.

COMPOUND: {compound}
DRUG CLASS: {drug_class}
TARGET: {target}
INDICATION: {indication}

MISSING/PARTIAL STUDIES (from IND gap assessment):
{gaps}

OUTPUT FORMAT — each study on its own line, then a summary line.

Study schema:
{{"type":"study","id":"S01","name":"<concise name>","subsection":"<Section 8 subsection>","regulatory_ref":"<exact CFR/ICH clause this study satisfies, e.g. 'ICH S7A §4.4 · Safety Pharmacology Core Battery — CNS'>","phase1_required":<bool>,"track":"<A|B|C>","track_name":"<In Vitro & Assays|Safety Pharmacology|GLP Toxicology>","week_start":<int>,"week_end":<int>,"cost_low":<int>,"cost_high":<int>,"cost_basis":"<1 sentence: which CRO tier and what drives the range for this study type>","failure_likelihood":"<low|medium|high>","failure_pct":<int 0-100>,"failure_rationale":"<1 sentence specific to this compound class — why it fails and what the consequence is>","fda_tier":"<phase1_blocking|conditional|phase2_defer>","depends_on":["S01",...],"compound_flag":"<specific design warning or null>","recommendation":"<1 sentence on key design consideration>","cost_saving_tips":["<tip 1: specific, verifiable, cite the mechanism e.g. 'Multiplex PPB with MDR1 assay at same CRO — Cyprotex panel pricing reduces per-assay cost ~30%'>","<tip 2 — optional>"],"context":"<1 sentence: either 'Extends [EXPXX] from source paper ([assay type] already performed; this adds GLP confirmation)' OR 'New study — no [assay type] data present in source paper' OR 'Conditional follow-up to [SXX]: only required if [trigger condition]'>","pubmed_query":"<5-8 PubMed search terms for finding published precedent studies of this exact assay in this compound class, e.g. 'hERG inhibition PDE4 inhibitor safety pharmacology preclinical'>"}}

Summary (last line):
{{"type":"summary","total_weeks_to_phase1":<int>,"cost_low":<int>,"cost_high":<int>,"critical_path":["S01",...],"phase1_gate_week":<int>,"phase2_studies":["S08",...],"compound_assessment":"<1 sentence>","highlights":[{{"label":"<6 words max>","detail":"<1 tight sentence>","tone":"<positive|warning|neutral>"}},{{"label":"<6 words max>","detail":"<1 tight sentence>","tone":"<positive|warning|neutral>"}},{{"label":"<6 words max>","detail":"<1 tight sentence>","tone":"<positive|warning|neutral>"}}]}}

TRACKS:
- Track A (In Vitro & Assays): all in vitro work — hERG, CYP, genotoxicity, PPB. Start week 0, run in parallel.
- Track B (Safety Pharmacology): in vivo CNS FOB, respiratory plethysmography. Start week 4+.
- Track C (GLP Toxicology): dose-ranging → GLP repeat-dose tox. Start week 4+, critical path.

PHASE 1 REQUIREMENTS — mark phase1_required: true (ICH M3(R2)):
- GLP in vitro hERG (NOT in vivo telemetry unless hERG positive)
- GLP CNS safety pharmacology (FOB or modified Irwin, rat)
- GLP respiratory safety pharmacology (plethysmography, rat)
- GLP Ames test (5-strain, ±S9)
- GLP in vitro mammalian genotoxicity (MN or chromosomal aberration)
- Plasma protein binding (human, for DDI risk calc)
- CYP inhibition screening (HLM, reversible)
- CYP induction (cryopreserved human hepatocytes)
- GLP 2-week minimum repeat-dose oral toxicity (rodent, same route as clinical)

PHASE 2 / DEFER — mark phase1_required: false, fda_tier: phase2_defer:
- In vivo cardiovascular telemetry (defer if in vitro hERG clean)
- Non-rodent GLP repeat-dose tox
- Transporter screening (P-gp, BCRP, OATPs)
- Mechanism-based CYP inhibition / TDI (conditional on reversible IC50 result)
- Metabolite profiling / mass balance

DEPENDENCIES (depends_on field):
- PPB must complete before full DDI risk assessment (R-value calc)
- Dose-ranging pilot must complete before GLP repeat-dose tox (sets dose levels)
- GLP Ames: if result is positive or equivocal → triggers in vivo micronucleus (add as conditional study)
- GLP hERG: if positive → in vivo CV telemetry becomes phase1_blocking

CONDITIONAL STUDY TIMING — CRITICAL:
- A study with fda_tier: "conditional" cannot start at week 0. It is conditional precisely because you must wait for the trigger study's result before deciding to run it.
- week_start for a conditional study must be ≥ week_end of its trigger dependency + 1 (minimum 1 week decision time after result).
- Example: If in vivo micronucleus is conditional on Ames result (Ames ends week 10), set week_start ≥ 11. Do NOT set week_start: 0 for any conditional study.

COST REFERENCE (include CRO setup + study + report):
- Plasma protein binding (human): $5k–$12k, 2–4 wks
- CYP inhibition HLM panel: $10k–$20k, 4–8 wks
- CYP induction hepatocytes: $15k–$30k, 6–10 wks
- GLP hERG (manual patch clamp): $25k–$45k, 8–12 wks
- GLP Ames (5-strain ±S9): $15k–$25k, 8–10 wks
- GLP in vitro MN or CA: $20k–$35k, 8–12 wks
- GLP in vivo micronucleus (if triggered): $45k–$80k, 12–16 wks
- GLP CNS FOB (rat, single oral dose): $35k–$70k, 10–14 wks
- GLP respiratory plethysmography (rat): $25k–$55k, 10–14 wks
- Dose-ranging pilot study: $15k–$35k, 6–10 wks
- GLP 2-week rat oral repeat-dose: $120k–$200k, 14–18 wks
- GLP 4-week rat oral (Phase 2): $180k–$300k, 18–24 wks
- In vivo CV telemetry (dog/minipig, Phase 2): $180k–$320k, 16–24 wks

COMPOUND-CLASS RULES — apply based on drug_class:
- Use your knowledge of the drug class to set realistic failure_pct
- Note any known species sensitivities, class liabilities, or design considerations in compound_flag
- If prior genotoxicity data shows equivocal signal: flag in vitro MN as likely triggered

INDICATION-SPECIFIC RULES — these are MANDATORY, not suggestions:

IF indication is Alzheimer's disease OR any CNS neurodegenerative condition:
1. ADD a CNS distribution / BBB penetration study (Track A, non-GLP, weeks 2–8, $15k–$30k):
   - Measure brain/plasma ratio and CSF exposure after oral dosing in rat
   - Without confirmed CNS penetration, the entire program rationale fails — this is the single highest-priority study for a CNS drug
   - If BBB penetration is confirmed low (<0.1 brain:plasma ratio), the program stops here
   - compound_flag must note: "BBB penetration unconfirmed — CNS distribution study is program-defining"
2. MODIFY the CNS FOB study to include COGNITIVE ENDPOINTS:
   - Standard Irwin/FOB is not sufficient for an Alzheimer's drug — add novel object recognition (NOR) or Y-maze test to assess potential cognitive impairment
   - Add to endpoints: "novel object recognition index, Y-maze spontaneous alternation"
   - failure_pct elevated to 25–35% for PDE4 inhibitors (GI emesis/nausea at CNS-effective doses is the primary class liability)
3. ADD a DDI note for CYP2D6: Alzheimer's patients are typically on donepezil (CYP2D6 substrate) — CYP2D6 inhibition must be explicitly screened
4. MODIFY Phase 1 protocol context: note that the clinical population may need to include MCI or prodromal AD patients (not healthy volunteers) for PD biomarker readout (CSF Aβ42/tau)
5. COST REFERENCE for CNS distribution study: rat tissue distribution (brain, CSF, plasma), LC-MS/MS analysis, 6–8 weeks, $15k–$30k at standard CRO

IF indication involves PDE4 inhibition (any indication):
- GI tolerability is the primary Phase 1 dose-limiting factor (nausea, emesis, diarrhea at therapeutic doses)
- compound_flag must note: "PDE4 class — GI tolerability is rate-limiting; dose titration schedule is critical for Phase 1 safety"
- failure_pct for GLP repeat-dose tox should be 20–30% (GI histopathology at high doses common)
- Recommend slow dose titration in Phase 1 (lower cohort dose increments than standard 2–3× escalation)

COST-SAVING TIPS — only include tips that are grounded in one of these mechanisms:
1. CRO panel/multiplex pricing (e.g. Cyprotex, Eurofins run cassette N-in-1 metabolism screens at lower per-compound cost)
2. Non-GLP for exploratory studies (pilot dose-ranging, preliminary PK) — GLP not required per ICH M3(R2) Section 6 for non-pivotal studies
3. Consolidating studies at a single CRO to unlock master service agreement discounts (typically 10–25% volume reduction)
4. Running Track A assays in batch with another program at the same site (shared setup costs)
5. Using automated patch clamp (QPatch/IonWorks) instead of manual for hERG screening — acceptable for non-GLP; saves ~40% vs manual GLP (GLP still requires manual confirmation)
6. Running CNS FOB and respiratory plethysmography in the same animals on the same day (ICH S7A §4.4 permits this) — eliminates a full animal cohort
7. Combining single-dose toxicity endpoints into the dose-ranging pilot (MTD assessment + TK in same study) — eliminates separate single-dose study
8. Outsourcing to CROs in lower-cost regions (e.g. WuXi AppTec, Pharmalegacy) for non-GLP work — GLP must still meet FDA inspection standards
9. Leveraging published literature for mechanism-of-action data to satisfy primary pharmacology subsection — acceptable if peer-reviewed and compound-specific enough

For each tip, name the specific savings mechanism. Do NOT invent discounts or cite specific dollar figures you cannot verify.

Output 8–14 studies total. Consolidate related gaps into logical studies (e.g., multiple CYP gaps = 2 studies max).

HIGHLIGHTS — write exactly 3, covering different dimensions:
1. Biggest risk or bottleneck (e.g. high-failure study on critical path, known class liability) — tone: "warning" if serious, "neutral" if manageable
2. Biggest opportunity or strength (e.g. existing data that reduces workload, cost-saving consolidation possible) — tone: "positive"
3. Timeline or cost reality check (e.g. what drives the week count, what the cost range depends on) — tone: "neutral"
Keep label under 6 words, detail under 20 words, specific to this compound not generic."""


def extract_gaps(ind_map: dict) -> list[dict]:
    gaps = []
    for section in ind_map.get("sections", []):
        if section.get("number") == 8:
            for sub in section.get("subsections", []):
                if sub.get("status") in ["missing", "partial"]:
                    for req in sub.get("missing_experiments", []):
                        # req may be a string or an object {study, rationale, priority}
                        if isinstance(req, dict):
                            req_text = req.get("study") or req.get("description") or req.get("requirement") or str(req)
                        else:
                            req_text = str(req)
                        gaps.append({
                            "subsection": sub["name"],
                            "status": sub["status"],
                            "regulatory_ref": sub.get("regulatory_ref", ""),
                            "requirement": req_text,
                        })
    return gaps


async def stream_timeline(gaps: list[dict], compound: str, drug_class: str,
                          target: str, indication: str):
    """Stream study JSON objects one at a time, then a summary object."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = TIMELINE_PROMPT.format(
        compound=compound,
        drug_class=drug_class,
        target=target,
        indication=indication,
        gaps=json.dumps(gaps, indent=2),
    )

    depth = 0
    obj = ""
    in_obj = False
    all_studies = []
    summary = {}

    async with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
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
                            if parsed.get("type") == "study":
                                all_studies.append(parsed)
                                yield parsed
                            elif parsed.get("type") == "summary":
                                summary = parsed
                                yield parsed
                        except json.JSONDecodeError:
                            pass
                        obj = ""
                        in_obj = False
                elif in_obj:
                    obj += char

    result = {"studies": all_studies, "summary": summary}
    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_DIR / "timeline.json", "w") as f:
        json.dump(result, f, indent=2)
