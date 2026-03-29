import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from models import PaperResponse

load_dotenv()

PIPELINE_STATE_DIR = Path(__file__).parent.parent / "pipeline-state"
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are an expert pharmacologist and regulatory scientist specializing in IND (Investigational New Drug) applications. Your task is to extract all experimental data from a scientific paper.

Extract and return a JSON object with this exact schema:
{
  "compound": "<name of the drug/compound being studied>",
  "experiments": [
    {
      "id": "<unique identifier, e.g. EXP-1>",
      "type": "<experiment type: in vitro, in vivo, ex vivo, clinical, etc.>",
      "model": "<experimental model: cell line name, animal model, human, etc.>",
      "species": "<species: human, rat, mouse, rabbit, dog, monkey, etc. — null if not applicable>",
      "route": "<administration route: oral, IV, IP, SC, topical, etc. — null if not applicable>",
      "dose": "<dose or dose range with units — null if not mentioned>",
      "n": "<sample size — null if not mentioned>",
      "glp_status": "<GLP, non-GLP, or unknown>",
      "endpoints": "<measured endpoints as a descriptive string>",
      "results": "<key findings including quantitative results and statistics>",
      "limitations": "<noted limitations or caveats — null if none mentioned>"
    }
  ],
  "claims": ["<major scientific claim 1>", "<major scientific claim 2>"]
}

Rules:
- Extract EVERY experiment mentioned, including preliminary or pilot studies
- claims should be major scientific assertions supported by data (efficacy, safety, mechanism of action)
- Be precise with numbers, units, and statistical values (p-values, confidence intervals, fold-changes)
- If a field is unknown or not mentioned, use null
- Return only valid JSON with no commentary or markdown fences"""


def _build_paper_text(paper: PaperResponse) -> str:
    """Assemble all available paper text into a single string for Claude."""
    parts = []

    if paper.title:
        parts.append(f"TITLE:\n{paper.title}")

    if paper.abstract:
        parts.append(f"ABSTRACT:\n{paper.abstract}")

    if paper.introduction:
        parts.append(f"INTRODUCTION:\n{paper.introduction}")

    if paper.methods_section:
        parts.append(f"METHODS:\n{paper.methods_section}")

    if paper.results_section:
        parts.append(f"RESULTS:\n{paper.results_section}")

    if paper.discussion_section:
        parts.append(f"DISCUSSION:\n{paper.discussion_section}")

    # Include any sections not captured by the classified fields
    classified_headings = {"introduction", "methods", "materials", "experimental", "results", "discussion", "conclusion"}
    for sec in paper.full_body_sections:
        if not any(kw in sec.heading.lower() for kw in classified_headings):
            parts.append(f"{sec.heading.upper()}:\n{sec.text}")

    # Figure captions often contain quantitative results
    if paper.figure_captions:
        fig_texts = []
        for fig in paper.figure_captions:
            caption = " ".join(filter(None, [fig.caption_title, fig.caption_text]))
            if caption:
                fig_texts.append(f"{fig.label}: {caption}")
        if fig_texts:
            parts.append("FIGURE CAPTIONS:\n" + "\n".join(fig_texts))

    # Tables
    if paper.tables:
        table_texts = []
        for tbl in paper.tables:
            lines = [f"{tbl.label}: {tbl.caption}"]
            if tbl.headers:
                lines.append("  Headers: " + " | ".join(tbl.headers))
            for row in tbl.data[:20]:  # cap to avoid token overflow
                lines.append("  " + " | ".join(row))
            table_texts.append("\n".join(lines))
        parts.append("TABLES:\n" + "\n\n".join(table_texts))

    return "\n\n".join(parts)


async def extract_experiments(paper: PaperResponse) -> dict:
    """Call Claude to extract experimental data from paper text, write to pipeline-state."""
    paper_text = _build_paper_text(paper)

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = await client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Extract all experimental data from the following paper:\n\n{paper_text}",
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if Claude wrapped the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    extracted = json.loads(raw)

    # Write to pipeline-state
    PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PIPELINE_STATE_DIR / "paper_data.json"
    with open(out_path, "w") as f:
        json.dump(extracted, f, indent=2)

    # Console summary
    n_experiments = len(extracted.get("experiments", []))
    n_claims = len(extracted.get("claims", []))
    compound = extracted.get("compound", "unknown")
    print(f"[stage2] Compound: {compound}")
    print(f"[stage2] Experiments extracted: {n_experiments}")
    print(f"[stage2] Claims identified:     {n_claims}")
    print(f"[stage2] Written to: {out_path}")

    return extracted
