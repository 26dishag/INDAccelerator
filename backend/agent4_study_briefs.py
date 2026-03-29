import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a GLP study design expert and CRO liaison with 20 years of preclinical drug development experience.
Output ONLY newline-delimited JSON objects (NDJSON). Each brief as a separate top-level JSON object.
No wrapper arrays, no markdown, no commentary."""

BRIEF_PROMPT = """Generate a concise study design brief for each study below.
These briefs will be sent to CROs to get quotes and initiate GLP protocol development.

COMPOUND: {compound}
DRUG CLASS: {drug_class}
TARGET: {target}
INDICATION: {indication}

PAPER EXPERIMENTS (what has already been done):
{paper_experiments}

TIMELINE STUDIES (what needs to be commissioned):
{studies}

For each study in the timeline, output ONE JSON object:

{{"type":"brief","study_id":"<S01>","study_name":"<name>","regulatory_basis":"<exact ICH/CFR clause>","objective":"<one sentence: what this study determines and why FDA requires it>","species":"<e.g. Sprague-Dawley rat>","strain_sex":"<e.g. Male and female, n=10/sex>","route":"<e.g. Oral gavage — matching proposed clinical route>","dose_levels":"<e.g. 0 (vehicle), 10, 30, 100 mg/kg — justified below>","dose_rationale":"<1-2 sentences: how doses derived from paper data, e.g. NOAEL from 28-day study, MTD from dose-ranging>","duration":"<e.g. Single dose + 7-day observation>","primary_endpoints":["<endpoint 1>","<endpoint 2>"],"secondary_endpoints":["<endpoint 1>"],"glp_required":<bool>,"acceptance_criteria":"<1 sentence: what a passing result looks like, e.g. no treatment-related CNS effects at doses ≥10× proposed clinical dose>","cro_minimum_requirements":["<e.g. FDA-compliant GLP facility>","<e.g. Accredited plethysmography equipment>"],"data_deliverables":["<e.g. Final GLP study report (21 CFR Part 58)>","<e.g. Raw data on CD>"],"paper_connection":"<1 sentence: how this study connects to or extends existing paper experiments — cite specific EXP IDs if relevant>","fail_consequence":"<1 sentence: what a failure means for the IND — program stop, protocol change, or additional study triggered>","cro_notes":"<1 sentence: any special requirements or scheduling notes for the CRO>"}}

RULES:
- Use paper experiment data to justify dose levels wherever possible
- If the paper has in vivo MTD or dose-response data, reference it for dose selection
- GLP required = true for: safety pharmacology core battery, genotoxicity, repeat-dose tox
- GLP required = false for: dose-ranging pilots, exploratory PK, non-pivotal screening assays
- Species: use rat for most studies unless a specific reason requires dog/minipig (CV telemetry only)
- Route: always match the proposed clinical route (oral for oral drugs)
- acceptance_criteria: frame in terms of multiples of expected clinical dose (e.g. clean at 10× clinical)
- dose_rationale: be specific — if paper shows IC50=50nM and free fraction=0.1, say so
- cro_notes: note if this study can be combined with another (e.g. CNS FOB + respiratory same day per ICH S7A §4.4)

INDICATION-SPECIFIC DESIGN RULES:
- If indication is Alzheimer's or CNS neurodegeneration:
  * CNS distribution study: dose_levels must include 3 doses bracketing expected therapeutic exposure; endpoints must include brain tissue, CSF, and plasma LC-MS/MS at multiple timepoints; acceptance_criteria = "brain:plasma ratio ≥ 0.3 at Cmax required to proceed"
  * CNS FOB study: primary_endpoints must include novel object recognition (NOR) index and Y-maze spontaneous alternation % in addition to standard Irwin battery
  * CYP DDI brief: explicitly call out CYP2D6 (donepezil is a CYP2D6 substrate — interaction risk in target population)
- If drug class is PDE4 inhibitor:
  * All repeat-dose tox briefs: primary_endpoints must include GI histopathology (stomach, duodenum, jejunum), body weight, and food consumption — GI tolerability is the class dose-limiting factor
  * dose_rationale must note: "GI tolerability expected to limit maximum tolerated dose; dose titration recommended"
  * fail_consequence for repeat-dose tox: "GI pathology at therapeutic multiples may require dose titration or formulation change before Phase 1"

Output one JSON brief per study. Order them by study_id (S01, S02...).
"""


async def stream_study_briefs(studies: list[dict], paper_experiments: list[dict],
                               compound: str, drug_class: str, target: str, indication: str):
    """Stream one study design brief per study."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = BRIEF_PROMPT.format(
        compound=compound,
        drug_class=drug_class,
        target=target,
        indication=indication,
        paper_experiments=json.dumps(paper_experiments, indent=2),
        studies=json.dumps(studies, indent=2),
    )

    depth = 0
    obj = ""
    in_obj = False
    all_briefs = []

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
