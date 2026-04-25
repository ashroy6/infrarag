from __future__ import annotations

import re
from typing import Any

from app.embedding_service import get_embedding
from app.qdrant_client import search
from app.reranker import rerank_hits


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", " and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "about", "into", "over", "under", "than", "how", "why", "when", "where",
    "can", "could", "should", "would", "will", "may", "might", "tell", "me",
    "explain", "give", "show", "summarise", "summarize"
}

_PAGE_RANGE_RE = re.compile(
    r"\bpages?\s+(\d+)\s*(?:-|to)\s*(\d+)\b|\bpage\s+(\d+)\b",
    re.IGNORECASE,
)

UK_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-/.]+", normalized)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def _extract_page_range(query: str) -> tuple[int | None, int | None]:
    match = _PAGE_RANGE_RE.search(query or "")
    if not match:
        return None, None

    if match.group(1) and match.group(2):
        start = int(match.group(1))
        end = int(match.group(2))
        return min(start, end), max(start, end)

    if match.group(3):
        page = int(match.group(3))
        return page, page

    return None, None


def _detect_query_intent(query: str) -> str:
    q = _normalize_text(query)

    contact_terms = {
        "address",
        "registered office",
        "office address",
        "company address",
        "postcode",
        "post code",
        "company number",
        "registration number",
        "registered in",
        "registered company",
        "contact address",
        "office location",
    }

    repo_terms = {
        "repo",
        "repository",
        "project",
        "overview",
        "purpose",
        "summary",
        "readme",
        "architecture",
        "workflow",
        "pipeline",
        "what does this repo do",
        "what is this repo",
        "what does this project do",
    }

    code_terms = {
        "function",
        "class",
        "method",
        "code",
        "script",
        "file",
        "module",
        "bug",
        "logic",
        "implementation",
        "what does this code do",
        "why does",
    }

    cicd_terms = {
        "ci",
        "cd",
        "cicd",
        "github actions",
        "workflow",
        "pipeline",
        "deploy",
        "deployment",
        "build",
        "test",
        "release",
    }

    infra_terms = {
        "terraform",
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "helm",
        "vpc",
        "iam",
        "s3",
        "eks",
        "rds",
        "infra",
        "infrastructure",
    }

    document_terms = {
        "book",
        "chapter",
        "pages",
        "page",
        "foreword",
        "contents",
        "author",
        "translator",
        "appendix",
        "section",
        "document",
        "pdf",
    }

    if any(term in q for term in contact_terms):
        return "contact"
    if any(term in q for term in document_terms):
        return "document"
    if any(term in q for term in repo_terms):
        return "repo"
    if any(term in q for term in cicd_terms):
        return "cicd"
    if any(term in q for term in infra_terms):
        return "infra"
    if any(term in q for term in code_terms):
        return "code"

    return "general"


def _extract_query_terms(query: str) -> list[str]:
    raw_terms = _tokenize(query)
    expanded = list(raw_terms)

    q = _normalize_text(query)
    intent = _detect_query_intent(query)

    if intent == "contact":
        expanded.extend(
            [
                "address",
                "registered",
                "office",
                "registered office",
                "company",
                "company number",
                "registration number",
                "registered company",
                "registered in",
                "postcode",
                "post code",
                "contact",
                "location",
            ]
        )

    elif intent == "repo":
        expanded.extend(
            [
                "readme",
                "overview",
                "purpose",
                "project",
                "repository",
                "workflow",
                "pipeline",
                "architecture",
            ]
        )

    elif intent == "cicd":
        expanded.extend(
            [
                "ci",
                "cd",
                "pipeline",
                "workflow",
                "github",
                "actions",
                "build",
                "test",
                "deploy",
                "release",
            ]
        )

    elif intent == "infra":
        expanded.extend(
            [
                "terraform",
                "cloud",
                "aws",
                "azure",
                "gcp",
                "kubernetes",
                "infrastructure",
                "module",
                "resource",
                "deployment",
                "cluster",
            ]
        )

    elif intent == "code":
        expanded.extend(
            [
                "function",
                "class",
                "module",
                "script",
                "logic",
                "implementation",
            ]
        )

    elif intent == "document":
        expanded.extend(
            [
                "document",
                "book",
                "chapter",
                "pages",
                "page",
                "section",
                "contents",
                "summary",
                "author",
                "appendix",
            ]
        )

    if "readme" in q:
        expanded.extend(["overview", "purpose", "summary"])

    return _unique_preserve_order(expanded)


def _keyword_overlap_score(query_terms: list[str], text: str, source: str) -> float:
    haystack_tokens = set(_tokenize(f"{source}\n{text}"))
    query_token_set = set(query_terms)

    matched_terms = query_token_set.intersection(haystack_tokens)
    if not matched_terms:
        return 0.0

    coverage = len(matched_terms) / max(len(query_token_set), 1)
    dense_bonus = min(len(matched_terms) * 0.03, 0.18)
    return coverage + dense_bonus


def _source_boost(hit: dict[str, Any], intent: str, query_terms: list[str]) -> float:
    source = (hit.get("source", "") or "").lower()
    source_type = (hit.get("source_type", "") or "").lower()
    file_type = (hit.get("file_type", "") or "").lower()

    boost = 0.0

    is_readme = "readme" in source
    is_docs = source.endswith(".md") or source.endswith(".rst") or source.endswith(".txt") or "/docs/" in source
    is_ci = ".github/workflows/" in source or source.endswith(".yml") or source.endswith(".yaml")
    is_code = source.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rs"))
    is_infra = (
        source.endswith(".tf")
        or source.endswith(".tfvars")
        or "terraform" in source
        or "helm" in source
        or "k8s" in source
        or "kubernetes" in source
    )
    is_test = "/tests/" in source or "test_" in source or source.endswith("_test.py")
    is_pdf = file_type == ".pdf"
    is_upload = source_type == "upload"

    if intent == "contact":
        if is_pdf:
            boost += 0.10
        if is_upload:
            boost += 0.05
        if file_type in {".pdf", ".md", ".txt"}:
            boost += 0.05

    elif intent == "repo":
        if is_readme:
            boost += 0.20
        if is_docs:
            boost += 0.10
        if is_ci:
            boost += 0.02
        if is_test:
            boost -= 0.04

    elif intent == "cicd":
        if is_ci:
            boost += 0.20
        if is_readme:
            boost += 0.04
        if is_docs:
            boost += 0.03
        if is_code:
            boost += 0.02

    elif intent == "infra":
        if is_infra:
            boost += 0.20
        if is_docs:
            boost += 0.05
        if is_readme:
            boost += 0.03

    elif intent == "code":
        if is_code:
            boost += 0.18
        if is_test:
            boost += 0.03
        if is_readme:
            boost -= 0.02

    elif intent == "document":
        if is_pdf:
            boost += 0.12
        if is_upload:
            boost += 0.05

    else:
        if is_readme:
            boost += 0.06
        if is_docs:
            boost += 0.04

    for term in query_terms:
        if term and term in source:
            boost += 0.01

    return boost


def _section_boost(text: str, intent: str) -> float:
    t = (text or "").lower()
    boost = 0.0

    repo_section_terms = [
        "project purpose",
        "overview",
        "what it does",
        "architecture",
        "workflow",
        "pipeline",
        "summary",
        "introduction",
    ]

    cicd_terms = [
        "github actions",
        "workflow",
        "build",
        "test",
        "deploy",
        "deployment",
        "job",
        "stage",
    ]

    infra_terms = [
        "terraform",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "helm",
        "module",
        "resource",
        "cluster",
        "infrastructure",
    ]

    document_terms = [
        "contents",
        "foreword",
        "chapter",
        "section",
        "appendix",
        "introduction",
    ]

    contact_terms = [
        "registered office",
        "company number",
        "registration number",
        "registered in",
        "office address",
        "company address",
        "postcode",
        "post code",
        "contact address",
    ]

    if intent == "contact":
        if any(term in t for term in contact_terms):
            boost += 0.30
        if UK_POSTCODE_RE.search(text or ""):
            boost += 0.18

    elif intent == "repo" and any(term in t for term in repo_section_terms):
        boost += 0.10

    elif intent == "cicd" and any(term in t for term in cicd_terms):
        boost += 0.10

    elif intent == "infra" and any(term in t for term in infra_terms):
        boost += 0.10

    elif intent == "document" and any(term in t for term in document_terms):
        boost += 0.10

    return boost


def _normalize_vector_score(score: float) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _hybrid_score(hit: dict[str, Any], query_terms: list[str], intent: str) -> float:
    vector_score = _normalize_vector_score(hit.get("score", 0.0))
    source = hit.get("source", "") or ""
    text = hit.get("text", "") or ""

    keyword_score = _keyword_overlap_score(query_terms, text, source)
    file_boost = _source_boost(hit, intent, query_terms)
    section_boost = _section_boost(text, intent)

    raw_hybrid = (
        (0.70 * vector_score)
        + (0.20 * min(keyword_score, 1.0))
        + file_boost
        + section_boost
    )

    if intent in {"repo", "document", "contact"} and len(text.split()) < 15:
        raw_hybrid -= 0.03

    return round(max(0.0, min(raw_hybrid, 0.9999)), 6)


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for hit in hits:
        key = (
            hit.get("source", ""),
            hit.get("chunk_index", -1),
        )
        if key in seen:
            continue

        seen.add(key)
        deduped.append(hit)

    return deduped


def retrieve_context(
    query: str,
    limit: int = 5,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))
    inferred_page_start, inferred_page_end = _extract_page_range(query)

    if page_start is None:
        page_start = inferred_page_start
    if page_end is None:
        page_end = inferred_page_end

    intent = _detect_query_intent(query)
    query_terms = _extract_query_terms(query)

    if (
        page_start is not None
        and page_end is not None
        and source_type is None
        and file_type is None
        and source_id is None
        and source is None
    ):
        source_type = "upload"
        file_type = ".pdf"

    query_vector = get_embedding(query)
    candidate_limit = max(safe_limit * 5, 30)

    initial_hits = search(
        query_vector=query_vector,
        limit=candidate_limit,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    initial_hits = _dedupe_hits(initial_hits)

    reranked = []
    for hit in initial_hits:
        hit_copy = dict(hit)
        hit_copy["vector_score"] = _normalize_vector_score(hit.get("score", 0.0))
        hit_copy["hybrid_score"] = _hybrid_score(hit_copy, query_terms, intent)
        hit_copy["intent"] = intent
        reranked.append(hit_copy)

    reranked.sort(
        key=lambda x: (
            x.get("hybrid_score", 0.0),
            x.get("vector_score", 0.0),
            -len((x.get("text", "") or "").split()),
        ),
        reverse=True,
    )

    hybrid_hits = []
    for hit in reranked:
        item = dict(hit)
        item["score"] = item["hybrid_score"]
        hybrid_hits.append(item)

    final_hits = rerank_hits(
        question=query,
        hits=hybrid_hits,
        top_n=safe_limit,
    )

    return final_hits
