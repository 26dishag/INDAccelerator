from pydantic import BaseModel
from typing import List, Optional


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class BodySection(BaseModel):
    heading: str
    text: str


class FigureCaption(BaseModel):
    id: str
    label: str
    caption_title: str
    caption_text: str


class TableEntry(BaseModel):
    id: str
    label: str
    caption: str
    note: str
    headers: List[str] = []
    data: List[List[str]] = []


class Reference(BaseModel):
    pmid: Optional[str] = None
    doi: Optional[str] = None
    citation_text: str


class FetchSources(BaseModel):
    pmc_full_text: str
    sections_found: List[str]
    figures_found: int
    tables_found: int
    references_found: int


class PaperResponse(BaseModel):
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    authors: List[Author] = []
    keywords: List[str] = []
    abstract: Optional[str] = None
    full_text_available: bool = False
    full_body_sections: List[BodySection] = []
    introduction: Optional[str] = None
    methods_section: Optional[str] = None
    results_section: Optional[str] = None
    discussion_section: Optional[str] = None
    figure_captions: List[FigureCaption] = []
    tables: List[TableEntry] = []
    references: List[Reference] = []
    mesh_terms: List[str] = []
    fetch_sources: Optional[FetchSources] = None
    error: Optional[str] = None
    warnings: List[str] = []
