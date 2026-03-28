import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
import httpx

from models import (
    Author, BodySection, FigureCaption, TableEntry, Reference,
    FetchSources, PaperResponse
)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_PARAMS = "tool=INDAccelerator&email=ind@accelerator.com"


# ---------------------------------------------------------------------------
# Step 1 — parse the input string
# ---------------------------------------------------------------------------

def parse_input(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (pmid, pmcid). One of them may be None."""
    raw = raw.strip()

    # PMC URL: ncbi.nlm.nih.gov/pmc/articles/PMC12345678
    m = re.search(r'pmc/articles/PMC(\d+)', raw, re.IGNORECASE)
    if m:
        return None, m.group(1)

    # PubMed URL: pubmed.ncbi.nlm.nih.gov/41585874
    m = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', raw)
    if m:
        return m.group(1), None

    # DOI URL — need to resolve via esummary later; treat as PMID lookup by DOI
    # We'll handle this by returning the DOI string keyed specially
    m = re.match(r'https?://doi\.org/(.+)', raw)
    if m:
        return None, None  # caller handles DOI separately — see fetch_paper

    # Bare PMCID
    m = re.match(r'PMC(\d+)$', raw, re.IGNORECASE)
    if m:
        return None, m.group(1)

    # Bare PMID (all digits)
    if re.match(r'^\d+$', raw):
        return raw, None

    return None, None


def extract_doi_from_input(raw: str) -> Optional[str]:
    m = re.match(r'https?://doi\.org/(.+)', raw.strip())
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Step 2 — resolve PMID → PMCID via esummary
# ---------------------------------------------------------------------------

async def resolve_pmcid_from_pmid(pmid: str, client: httpx.AsyncClient) -> Optional[str]:
    url = (
        f"{EUTILS_BASE}/esummary.fcgi"
        f"?db=pubmed&id={pmid}&retmode=json&{TOOL_PARAMS}"
    )
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    article_ids = data.get("result", {}).get(pmid, {}).get("articleids", [])
    for aid in article_ids:
        if aid.get("idtype") == "pmc":
            val = aid.get("value", "")
            # strip "PMC" prefix if present
            return re.sub(r'^PMC', '', val, flags=re.IGNORECASE)
    return None


async def resolve_pmid_from_doi(doi: str, client: httpx.AsyncClient) -> Optional[str]:
    """Use esearch to find PMID for a DOI."""
    url = (
        f"{EUTILS_BASE}/esearch.fcgi"
        f"?db=pubmed&term={doi}[doi]&retmode=json&{TOOL_PARAMS}"
    )
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# Step 3 — fetch full PMC XML
# ---------------------------------------------------------------------------

async def fetch_pmc_xml(pmcid: str, client: httpx.AsyncClient) -> str:
    url = (
        f"{EUTILS_BASE}/efetch.fcgi"
        f"?db=pmc&id={pmcid}&rettype=xml&retmode=xml&{TOOL_PARAMS}"
    )
    resp = await client.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


async def fetch_pubmed_abstract_xml(pmid: str, client: httpx.AsyncClient) -> str:
    url = (
        f"{EUTILS_BASE}/efetch.fcgi"
        f"?db=pubmed&id={pmid}&rettype=abstract&retmode=xml&{TOOL_PARAMS}"
    )
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Step 4 — XML parsing helpers
# ---------------------------------------------------------------------------

def _text(el: Optional[ET.Element]) -> str:
    """Recursively collect all text from an element."""
    if el is None:
        return ""
    parts = []
    if el.text:
        parts.append(el.text.strip())
    for child in el:
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def _find_ns(root: ET.Element, tag: str):
    """Find ignoring namespace prefix."""
    # Try plain tag first
    el = root.find(".//" + tag)
    if el is not None:
        return el
    # Try with wildcard namespace
    el = root.find(".//{*}" + tag)
    return el


def _findall_ns(root: ET.Element, tag: str):
    results = root.findall(".//" + tag)
    if not results:
        results = root.findall(".//{*}" + tag)
    return results


def parse_pmc_xml(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)

    result = {}

    # -- title --
    title_el = _find_ns(root, "article-title")
    result["title"] = _text(title_el) if title_el is not None else None

    # -- abstract --
    abstract_parts = []
    for abs_el in _findall_ns(root, "abstract"):
        for p in abs_el.findall(".//p") or abs_el.findall(".//{*}p"):
            t = _text(p)
            if t:
                abstract_parts.append(t)
    result["abstract"] = " ".join(abstract_parts) if abstract_parts else None

    # -- authors + affiliations --
    # First collect affiliations keyed by id
    aff_map = {}
    for aff in _findall_ns(root, "aff"):
        aff_id = aff.get("id", "")
        aff_map[aff_id] = _text(aff)

    authors = []
    for contrib in _findall_ns(root, "contrib"):
        if contrib.get("contrib-type") != "author":
            continue
        surname_el = _find_ns(contrib, "surname")
        given_el = _find_ns(contrib, "given-names")
        surname = _text(surname_el) if surname_el is not None else ""
        given = _text(given_el) if given_el is not None else ""
        name = f"{given} {surname}".strip()
        # resolve affiliation
        aff_ref = contrib.find(".//xref[@ref-type='aff']")
        if aff_ref is None:
            aff_ref = contrib.find(".//{*}xref[@ref-type='aff']")
        aff_text = None
        if aff_ref is not None:
            rid = aff_ref.get("rid", "")
            aff_text = aff_map.get(rid)
        if name:
            authors.append(Author(name=name, affiliation=aff_text))
    result["authors"] = authors

    # -- journal --
    journal_el = _find_ns(root, "journal-title")
    result["journal"] = _text(journal_el) if journal_el is not None else None

    # -- year -- try epub first, then collection, then any pub-date
    year = None
    for pub_date in _findall_ns(root, "pub-date"):
        pub_type = pub_date.get("pub-type", pub_date.get("date-type", ""))
        yr_el = pub_date.find("year") or pub_date.find("{*}year")
        if yr_el is not None and pub_type in ("epub", "collection", "ppub"):
            year = _text(yr_el)
            if pub_type == "epub":
                break
    if not year:
        yr_el = _find_ns(root, "year")
        if yr_el is not None:
            year = _text(yr_el)
    result["year"] = year

    # -- IDs --
    result["doi"] = None
    result["pmid"] = None
    result["pmcid"] = None
    for aid in _findall_ns(root, "article-id"):
        id_type = aid.get("pub-id-type", "")
        val = _text(aid)
        if id_type == "doi":
            result["doi"] = val
        elif id_type == "pmid":
            result["pmid"] = val
        elif id_type == "pmc":
            result["pmcid"] = f"PMC{val}" if not val.startswith("PMC") else val

    # -- keywords --
    keywords = []
    for kwd in _findall_ns(root, "kwd"):
        t = _text(kwd)
        if t:
            keywords.append(t)
    result["keywords"] = keywords

    # -- body sections --
    body_el = _find_ns(root, "body")
    sections = []
    classified = {"introduction": None, "methods": None, "results": None, "discussion": None}

    if body_el is not None:
        # grab top-level <sec> elements
        top_secs = body_el.findall("sec") or body_el.findall("{*}sec")
        if not top_secs:
            top_secs = _findall_ns(body_el, "sec")

        for sec in top_secs:
            heading_el = sec.find("title") or sec.find("{*}title")
            heading = _text(heading_el) if heading_el is not None else "Untitled"

            # collect all text from <p> elements (including nested secs)
            paras = []
            for p in sec.iter():
                if p.tag in ("p", "{*}p") or p.tag.endswith("}p") or p.tag == "p":
                    t = _text(p)
                    if t:
                        paras.append(t)

            # deduplicate while preserving order
            seen = set()
            unique_paras = []
            for p in paras:
                if p not in seen:
                    seen.add(p)
                    unique_paras.append(p)

            text = " ".join(unique_paras)
            sections.append(BodySection(heading=heading, text=text))

            # classify
            h_lower = heading.lower()
            if any(k in h_lower for k in ("introduc",)):
                classified["introduction"] = text
            elif any(k in h_lower for k in ("method", "material", "experimental")):
                classified["methods"] = text
            elif any(k in h_lower for k in ("result",)):
                classified["results"] = text
            elif any(k in h_lower for k in ("discussion", "conclusion")):
                classified["discussion"] = text

    result["full_body_sections"] = sections
    result["introduction"] = classified["introduction"]
    result["methods_section"] = classified["methods"]
    result["results_section"] = classified["results"]
    result["discussion_section"] = classified["discussion"]

    # -- figures --
    figures = []
    for fig in _findall_ns(root, "fig"):
        fig_id = fig.get("id", "")
        label_el = fig.find("label") or fig.find("{*}label")
        label = _text(label_el) if label_el is not None else ""

        cap_el = fig.find("caption") or fig.find("{*}caption")
        cap_title = ""
        cap_text = ""
        if cap_el is not None:
            title_el = cap_el.find("title") or cap_el.find("{*}title")
            cap_title = _text(title_el) if title_el is not None else ""
            para_parts = []
            for p in (cap_el.findall("p") or cap_el.findall("{*}p")):
                t = _text(p)
                if t:
                    para_parts.append(t)
            cap_text = " ".join(para_parts)

        figures.append(FigureCaption(
            id=fig_id,
            label=label,
            caption_title=cap_title,
            caption_text=cap_text,
        ))
    result["figure_captions"] = figures

    # -- tables --
    tables = []
    for tw in _findall_ns(root, "table-wrap"):
        tw_id = tw.get("id", "")
        label_el = tw.find("label") or tw.find("{*}label")
        label = _text(label_el) if label_el is not None else ""

        cap_el = tw.find("caption") or tw.find("{*}caption")
        caption = ""
        if cap_el is not None:
            caption = _text(cap_el)

        note_el = tw.find("table-wrap-foot") or tw.find("{*}table-wrap-foot")
        note = _text(note_el) if note_el is not None else ""

        # Parse actual table cell data
        headers = []
        data = []
        table_el = tw.find(".//table") or tw.find(".//{*}table")
        if table_el is not None:
            # Headers from <thead>
            thead = table_el.find(".//thead") or table_el.find(".//{*}thead")
            if thead is not None:
                for tr in (thead.findall(".//tr") or thead.findall(".//{*}tr")):
                    row = [_text(cell) for cell in list(tr) if cell.tag in ("th", "td") or cell.tag.endswith("}th") or cell.tag.endswith("}td")]
                    if row:
                        headers = row
                        break  # use first header row only

            # Rows from <tbody>; fall back to all <tr> if no tbody
            tbody = table_el.find(".//tbody") or table_el.find(".//{*}tbody")
            row_source = tbody if tbody is not None else table_el
            for tr in (row_source.findall(".//tr") or row_source.findall(".//{*}tr")):
                # skip rows that are purely header-cell rows
                cells = list(tr)
                if all(c.tag == "th" or c.tag.endswith("}th") for c in cells):
                    if not headers:
                        headers = [_text(c) for c in cells]
                    continue
                row = [_text(cell) for cell in cells if cell.tag in ("th", "td") or cell.tag.endswith("}th") or cell.tag.endswith("}td")]
                if row:
                    data.append(row)

        tables.append(TableEntry(id=tw_id, label=label, caption=caption, note=note, headers=headers, data=data))
    result["tables"] = tables

    # -- references --
    references = []
    for ref in _findall_ns(root, "ref"):
        ref_pmid = None
        ref_doi = None
        for pub_id in (ref.findall(".//pub-id") or ref.findall(".//{*}pub-id")):
            id_type = pub_id.get("pub-id-type", "")
            val = _text(pub_id)
            if id_type == "pmid":
                ref_pmid = val
            elif id_type == "doi":
                ref_doi = val

        # citation text from mixed-citation or element-citation
        cite_el = (
            ref.find("mixed-citation") or ref.find("{*}mixed-citation") or
            ref.find(".//mixed-citation") or ref.find(".//{*}mixed-citation") or
            ref.find("element-citation") or ref.find("{*}element-citation") or
            ref.find(".//element-citation") or ref.find(".//{*}element-citation")
        )
        citation_text = _text(cite_el) if cite_el is not None else _text(ref)

        references.append(Reference(
            pmid=ref_pmid,
            doi=ref_doi,
            citation_text=citation_text,
        ))
    result["references"] = references

    return result


def parse_pubmed_abstract_xml(xml_text: str) -> dict:
    """Fallback: parse abstract + MeSH from PubMed efetch XML."""
    root = ET.fromstring(xml_text)
    result = {"abstract": None, "mesh_terms": [], "title": None}

    # abstract
    parts = []
    for p in _findall_ns(root, "AbstractText"):
        t = _text(p)
        if t:
            parts.append(t)
    result["abstract"] = " ".join(parts) if parts else None

    # title
    title_el = _find_ns(root, "ArticleTitle")
    result["title"] = _text(title_el) if title_el is not None else None

    # MeSH
    mesh = []
    for mh in _findall_ns(root, "DescriptorName"):
        t = _text(mh)
        if t:
            mesh.append(t)
    result["mesh_terms"] = mesh

    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def fetch_and_parse(raw_input: str) -> PaperResponse:
    warnings = []

    async with httpx.AsyncClient() as client:
        pmid, pmcid = parse_input(raw_input)
        doi_input = extract_doi_from_input(raw_input)

        # If DOI URL given, resolve to PMID first
        if doi_input and not pmid and not pmcid:
            pmid = await resolve_pmid_from_doi(doi_input, client)
            if not pmid:
                return PaperResponse(
                    error=f"Could not resolve DOI '{doi_input}' to a PubMed ID. "
                          "Paper may not be indexed in PubMed."
                )

        if not pmid and not pmcid:
            return PaperResponse(
                error=f"Could not parse a PMID or PMCID from input: '{raw_input}'. "
                      "Accepted formats: PubMed URL, PMC URL, DOI URL, bare PMID, bare PMCID."
            )

        # Resolve PMID → PMCID if we only have PMID
        if pmid and not pmcid:
            pmcid = await resolve_pmcid_from_pmid(pmid, client)

        # Attempt full PMC XML fetch
        full_text_available = False
        pmc_status = "not_attempted"
        parsed = {}

        if pmcid:
            try:
                xml_text = await fetch_pmc_xml(pmcid, client)
                parsed = parse_pmc_xml(xml_text)
                full_text_available = True
                pmc_status = "ok"
            except Exception as e:
                pmc_status = f"failed: {e}"
                warnings.append(f"PMC full text fetch failed ({pmc_status}). Falling back to abstract only.")

        # Fallback to abstract if no full text
        mesh_terms = []
        if not full_text_available:
            if not pmid:
                return PaperResponse(
                    pmcid=f"PMC{pmcid}" if pmcid else None,
                    full_text_available=False,
                    error="Paper is not available in PMC and no PMID found for abstract fallback.",
                    warnings=warnings,
                )
            try:
                abs_xml = await fetch_pubmed_abstract_xml(pmid, client)
                abs_data = parse_pubmed_abstract_xml(abs_xml)
                parsed["abstract"] = abs_data.get("abstract")
                parsed["title"] = abs_data.get("title")
                mesh_terms = abs_data.get("mesh_terms", [])
                warnings.append(
                    "Full text not available in PMC. Only abstract and MeSH terms retrieved. "
                    "All body sections, figures, and tables are missing."
                )
            except Exception as e:
                return PaperResponse(
                    pmid=pmid,
                    full_text_available=False,
                    error=f"Both PMC full text and PubMed abstract fetch failed: {e}",
                    warnings=warnings,
                )

        # Also fetch MeSH terms from PubMed if we have PMID (supplements full text)
        if full_text_available and pmid:
            try:
                abs_xml = await fetch_pubmed_abstract_xml(pmid, client)
                abs_data = parse_pubmed_abstract_xml(abs_xml)
                mesh_terms = abs_data.get("mesh_terms", [])
            except Exception:
                warnings.append("Could not fetch MeSH terms from PubMed (non-fatal).")

        # Determine sections found
        sections_found = []
        body_sections: list[BodySection] = parsed.get("full_body_sections", [])
        if parsed.get("introduction"):
            sections_found.append("introduction")
        if parsed.get("methods_section"):
            sections_found.append("methods")
        if parsed.get("results_section"):
            sections_found.append("results")
        if parsed.get("discussion_section"):
            sections_found.append("discussion")

        figures: list[FigureCaption] = parsed.get("figure_captions", [])
        tables: list[TableEntry] = parsed.get("tables", [])
        references: list[Reference] = parsed.get("references", [])

        # Normalise IDs — prefer what we resolved over what's in the XML
        resolved_pmid = pmid or parsed.get("pmid")
        resolved_pmcid = (
            f"PMC{pmcid}" if pmcid and not str(pmcid).startswith("PMC") else pmcid
        ) or parsed.get("pmcid")

        return PaperResponse(
            pmid=resolved_pmid,
            pmcid=resolved_pmcid,
            doi=parsed.get("doi"),
            title=parsed.get("title"),
            journal=parsed.get("journal"),
            year=parsed.get("year"),
            authors=parsed.get("authors", []),
            keywords=parsed.get("keywords", []),
            abstract=parsed.get("abstract"),
            full_text_available=full_text_available,
            full_body_sections=body_sections,
            introduction=parsed.get("introduction"),
            methods_section=parsed.get("methods_section"),
            results_section=parsed.get("results_section"),
            discussion_section=parsed.get("discussion_section"),
            figure_captions=figures,
            tables=tables,
            references=references,
            mesh_terms=mesh_terms,
            fetch_sources=FetchSources(
                pmc_full_text=pmc_status,
                sections_found=sections_found,
                figures_found=len(figures),
                tables_found=len(tables),
                references_found=len(references),
            ),
            warnings=warnings,
        )
