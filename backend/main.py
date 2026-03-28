from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from stage1_ingest import fetch_and_parse
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
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
