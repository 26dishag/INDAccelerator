import json
import os
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv

from models import PaperResponse

load_dotenv()

MODEL = "claude-sonnet-4-5"


# ── Paper text builder ─────────────────────────────────────────────────────────

def _paper_text(paper: PaperResponse, limit: int | None = None) -> str:
    parts = []
    if paper.title:         parts.append(f"TITLE: {paper.title}")
    if paper.abstract:      parts.append(f"ABSTRACT: {paper.abstract}")
    if paper.introduction:  parts.append(f"INTRODUCTION: {paper.introduction[:2000]}")
    if paper.methods_section:   parts.append(f"METHODS: {paper.methods_section[:3000]}")
    if paper.results_section:   parts.append(f"RESULTS: {paper.results_section[:3000]}")
    if paper.discussion_section: parts.append(f"DISCUSSION: {paper.discussion_section[:2000]}")
    text = "\n\n".join(parts)
    return text[:limit] if limit else text


def _full_paper_text(paper: PaperResponse) -> str:
    parts = []
    if paper.title:    parts.append(f"TITLE: {paper.title}")
    if paper.abstract: parts.append(f"ABSTRACT: {paper.abstract}")
    if paper.methods_section:    parts.append(f"METHODS: {paper.methods_section}")
    if paper.results_section:    parts.append(f"RESULTS: {paper.results_section}")
    if paper.discussion_section: parts.append(f"DISCUSSION: {paper.discussion_section}")
    classified = {"introduction", "methods", "materials", "experimental", "results", "discussion", "conclusion"}
    for sec in paper.full_body_sections:
        if not any(k in sec.heading.lower() for k in classified):
            parts.append(f"{sec.heading.upper()}: {sec.text}")
    if paper.figure_captions:
        figs = " | ".join(
            f"{f.label}: {f.caption_title} {f.caption_text}"
            for f in paper.figure_captions if f.caption_text
        )
        if figs:
            parts.append(f"FIGURES: {figs}")
    return "\n\n".join(parts)


# ── Agent 1a: Compound profile ────────────────────────────────────────────────

async def get_profile(paper: PaperResponse) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    msg = await client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=(
            "You are a pharmaceutical regulatory expert. "
            "Extract the compound profile and recommend the best indication for an IND. "
            "Return ONLY valid JSON, no markdown fences."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"{_paper_text(paper, limit=6000)}\n\n"
                "Return exactly this JSON:\n"
                '{"compound":"<name>","drug_class":"<e.g. small molecule PDE4 inhibitor>",'
                '"primary_target":"<target>","mechanism":"<one sentence>",'
                '"indications_studied":["<list>"],'
                '"recommended_indication":"<best for IND>",'
                '"recommendation_rationale":"<2-3 sentences why>",'
                '"evidence_strength":"strong|moderate|weak",'
                '"development_stage":"preclinical|phase1_ready|phase2_ready",'
                '"alternative_indications":[{"indication":"...","rationale":"..."}]}'
            )
        }]
    )

    raw = msg.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return json.loads(raw)


# ── Agent 1b: Streaming experiment extractor ──────────────────────────────────

async def stream_experiments(paper: PaperResponse) -> AsyncGenerator[dict, None]:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    text = _full_paper_text(paper)

    depth = 0
    obj = ""
    in_obj = False

    async with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        system=(
            "Extract every experiment from this paper for IND planning. "
            "Output each as a standalone JSON object, one at a time. "
            "No array wrapper, no commentary, no markdown. "
            'Schema: {"id":"EXP-N","type":"in vitro|in vivo|ex vivo|in silico|clinical",'
            '"model":"...","species":"...","route":"...","dose":"...","n":"...",'
            '"glp_status":"GLP|non-GLP|unknown","endpoints":"...","results":"...","limitations":"..."}'
        ),
        messages=[{"role": "user", "content": f"Extract all experiments:\n\n{text}"}]
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
                            if "id" in parsed:
                                yield parsed
                        except json.JSONDecodeError:
                            pass
                        obj = ""
                        in_obj = False
                elif in_obj:
                    obj += char
