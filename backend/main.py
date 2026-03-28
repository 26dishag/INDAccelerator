import json
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from stage1_ingest import fetch_and_parse
from stage2_graph import build_graph_for_pmid, cache_path
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
    result = await fetch_and_parse(input.strip())

    # Cache paper for Stage 2 — write only if we have a PMID and no error
    if result.pmid and not result.error:
        p = cache_path(result.pmid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

    return result


@app.post("/api/build-graph")
async def build_graph_endpoint(body: dict):
    pmid = body.get("pmid", "").strip()
    if not pmid:
        raise HTTPException(status_code=400, detail="'pmid' is required")
    try:
        cyto = await build_graph_for_pmid(pmid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {e}")
    return cyto


@app.get("/health")
async def health():
    return {"status": "ok"}
