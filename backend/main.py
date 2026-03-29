import json
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from stage1_ingest import fetch_and_parse
from stage2_extract import extract_experiments, PIPELINE_STATE_DIR
from agent1_profiler import get_profile, stream_experiments
from agent2_ind_mapper import build_ind_map, stream_ind_map
from agent3_timeline import stream_timeline, extract_gaps
from agent4_study_briefs import stream_study_briefs
from agent5_ind_drafts import stream_ind_draft
from models import PaperResponse

app = FastAPI(title="INDAccelerator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/fetch-paper", response_model=PaperResponse)
async def fetch_paper(input: str = Query(..., description="PubMed URL, PMC URL, DOI URL, bare PMID, or bare PMCID")):
    if not input.strip():
        raise HTTPException(status_code=400, detail="'input' parameter is required")
    return await fetch_and_parse(input.strip())


@app.get("/api/extract")
async def extract(input: str = Query(..., description="PubMed URL, PMC URL, DOI URL, bare PMID, or bare PMCID")):
    if not input.strip():
        raise HTTPException(status_code=400, detail="'input' parameter is required")
    paper = await fetch_and_parse(input.strip())
    if paper.error:
        raise HTTPException(status_code=404, detail=paper.error)
    try:
        return await extract_experiments(paper)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream-analysis")
async def stream_analysis(paper: PaperResponse):
    async def generate():
        profile = {}
        all_experiments = []

        # Step 1: compound profile (fast, ~2s)
        try:
            profile = await get_profile(paper)
            yield f"data: {json.dumps({'type': 'profile', 'data': profile})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Step 2: stream experiments one by one
        async for exp in stream_experiments(paper):
            all_experiments.append(exp)
            yield f"data: {json.dumps({'type': 'experiment', 'data': exp})}\n\n"

        # Save to pipeline-state so downstream agents can read it
        PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
        paper_data = {
            "compound": profile.get("compound", ""),
            "experiments": all_experiments,
            "claims": [],
        }
        with open(PIPELINE_STATE_DIR / "paper_data.json", "w") as f:
            json.dump(paper_data, f, indent=2)
        with open(PIPELINE_STATE_DIR / "profile.json", "w") as f:
            json.dump(profile, f, indent=2)

        yield f"data: {json.dumps({'type': 'done', 'total': len(all_experiments)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/build-ind-map")
async def build_ind_map_endpoint():
    path = PIPELINE_STATE_DIR / "paper_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No paper data found. Run extraction first.")
    with open(path) as f:
        paper_data = json.load(f)

    async def generate():
        async for section in stream_ind_map(paper_data.get("experiments", [])):
            yield f"data: {json.dumps({'type': 'section', 'data': section})}\n\n"
        # Read summary from saved file
        summary = ""
        ind_path = PIPELINE_STATE_DIR / "ind_map.json"
        if ind_path.exists():
            with open(ind_path) as f:
                summary = json.load(f).get("summary", "")
        yield f"data: {json.dumps({'type': 'done', 'summary': summary})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ind-map")
async def get_ind_map():
    path = PIPELINE_STATE_DIR / "ind_map.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No IND map found. Run /api/build-ind-map first.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/paper-data")
async def get_paper_data():
    path = PIPELINE_STATE_DIR / "paper_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No extracted data found. Run /api/extract first.")
    with open(path) as f:
        return json.load(f)


@app.post("/api/build-timeline")
async def build_timeline_endpoint():
    ind_path = PIPELINE_STATE_DIR / "ind_map.json"
    if not ind_path.exists():
        raise HTTPException(status_code=404, detail="No IND map found. Run /api/build-ind-map first.")

    profile_path = PIPELINE_STATE_DIR / "profile.json"
    profile = {}
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
    else:
        # Fall back to paper_data for compound name
        pd_path = PIPELINE_STATE_DIR / "paper_data.json"
        if pd_path.exists():
            with open(pd_path) as f:
                pd = json.load(f)
            profile = {"compound": pd.get("compound", "Unknown"), "drug_class": "", "target": "", "indication": ""}

    with open(ind_path) as f:
        ind_map = json.load(f)

    gaps = extract_gaps(ind_map)
    if not gaps:
        raise HTTPException(status_code=400, detail="No gaps found in IND map.")

    async def generate():
        async for item in stream_timeline(
            gaps=gaps,
            compound=profile.get("compound", "Unknown"),
            drug_class=profile.get("drug_class", ""),
            target=profile.get("target", ""),
            indication=profile.get("indication", ""),
        ):
            yield f"data: {json.dumps({'type': item.get('type'), 'data': item})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/study-briefs")
async def build_study_briefs_endpoint():
    timeline_path = PIPELINE_STATE_DIR / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="No timeline found. Run /api/build-timeline first.")

    paper_path = PIPELINE_STATE_DIR / "paper_data.json"
    if not paper_path.exists():
        raise HTTPException(status_code=404, detail="No paper data found.")

    profile_path = PIPELINE_STATE_DIR / "profile.json"
    profile = {}
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)

    with open(timeline_path) as f:
        timeline = json.load(f)
    with open(paper_path) as f:
        paper_data = json.load(f)

    studies = timeline.get("studies", [])
    if not studies:
        raise HTTPException(status_code=400, detail="No studies found in timeline.")

    async def generate():
        async for brief in stream_study_briefs(
            studies=studies,
            paper_experiments=paper_data.get("experiments", []),
            compound=profile.get("compound", "Unknown"),
            drug_class=profile.get("drug_class", ""),
            target=profile.get("target", ""),
            indication=profile.get("indication", ""),
        ):
            yield f"data: {json.dumps({'type': 'brief', 'data': brief})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ind-draft/{section}")
async def get_ind_draft(section: int):
    if section < 1 or section > 10:
        raise HTTPException(status_code=400, detail="Section must be 1-10.")

    async def generate():
        async for chunk in stream_ind_draft(section):
            # Escape for SSE: replace newlines in chunk with sentinel, client decodes
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
