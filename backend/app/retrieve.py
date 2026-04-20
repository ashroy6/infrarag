from __future__ import annotations

import re
from typing import Any

from app.embedding_service import get_embedding
from app.qdrant_client import search


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "about", "into", "over", "under", "than", "how", "why", "when", "where",
    "can", "could", "should", "would", "will", "may", "might", "tell", "me",
    "explain", "give", "show"
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> list[str]:
    text = _normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-/.]+", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _detect_query_intent(query: str) -> str:
    q = _normalize_text(query)

    repo_terms = {
        "repo", "repository", "project", "overview", "purpose", "summary",
        "readme", "what does this repo do", "what is this repo", "architecture",
        "workflow", "pipeline", "what does this project do"
    }
    code_terms = {
        "function", "class", "method", "code", "script", "file", "module",
        "why does", "what does this code do", "bug", "logic", "implementation"
    }
    cicd_terms = {
        "ci", "cd", "cicd", "github actions", "workflow", "pipeline", "deploy",
        "build", "test", "release", ".github"
    }
    infra_terms = {
        "terraform", "aws", "kubernetes", "helm", "vpc", "iam", "s3", "eks",
        "rds", "infra", "infrastructure"
    }

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

    if intent == "repo":
        expanded.extend([
            "readme", "overview", "purpose", "project", "repository",
            "workflow", "pipeline", "architecture"
        ])
    elif intent == "cicd":
        expanded.extend([
            "ci", "cd", "pipeline", "workflow", "github", "actions",
            "build", "test", "deploy"
        ])
    elif intent == "infra":
        expanded.extend([
            "terraform", "aws", "kubernetes", "infrastructure", "module",
            "resource", "deployment", "cluster"
        ])
    elif intent == "code":
        expanded.extend([
            "function", "class", "module", "script", "logic", "implementation"
        ])

    if "readme" in q:
        expanded.extend(["overview", "purpose", "summary"])

    return _unique_preserve_order(expanded)


def _keyword_overlap_score(query_terms: list[str], text: str, source: str) -> float:
    haystack = f"{source}\n{text}".lower()
    matched_terms = [term for term in query_terms if term in haystack]

    if not matched_terms:
        return 0.0

    coverage = len(matched_terms) / max(len(query_terms), 1)
    dense_bonus = min(len(matched_terms) * 0.03, 0.18)
    return coverage + dense_bonus


def _source_boost(source: str, intent: str) -> float:
    s = (source or "").lower()

    boost = 0.0

    is_readme = "readme" in s
    is_docs = "/docs/" in s or s.endswith(".md") or s.endswith(".rst") or s.endswith(".txt")
    is_ci = ".github/workflows/" in s or s.endswith(".yml") or s.endswith(".yaml")
    is_code = s.endswith(".py") or s.endswith(".js") or s.endswith(".ts") or s.endswith(".go") or s.endswith(".java")
    is_infra = (
        s.endswith(".tf") or s.endswith(".tfvars") or "terraform" in s or
        "helm" in s or "k8s" in s or "kubernetes" in s
    )
    is_test = "/tests/" in s or "test_" in s or s.endswith("_test.py")

    if intent == "repo":
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

    else:
        if is_readme:
            boost += 0.06
        if is_docs:
            boost += 0.04

    return boost


def _section_boost(text: str, intent: str) -> float:
    t = (text or "").lower()

    boost = 0.0

    repo_section_terms = [
        "project purpose", "overview", "what it does", "architecture",
        "workflow", "pipeline", "summary", "introduction"
    ]
    cicd_terms = [
        "github actions", "workflow", "build", "test", "deploy", "job", "stage"
    ]
    infra_terms = [
        "terraform", "kubernetes", "aws", "helm", "module", "resource", "cluster"
    ]

    if intent == "repo" and any(term in t for term in repo_section_terms):
        boost += 0.10
    elif intent == "cicd" and any(term in t for term in cicd_terms):
        boost += 0.10
    elif intent == "infra" and any(term in t for term in infra_terms):
        boost += 0.10

    return boost


def _normalize_vector_score(score: float) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _hybrid_score(hit: dict[str, Any], query: str, query_terms: list[str], intent: str) -> float:
    vector_score = _normalize_vector_score(hit.get("score", 0.0))
    source = hit.get("source", "") or ""
    text = hit.get("text", "") or ""

    keyword_score = _keyword_overlap_score(query_terms, text, source)
    file_boost = _source_boost(source, intent)
    section_boost = _section_boost(text, intent)

    raw_hybrid = (
        (0.72 * vector_score) +
        (0.18 * min(keyword_score, 1.0)) +
        file_boost +
        section_boost
    )

    if intent == "repo" and len(text.split()) < 20:
        raw_hybrid -= 0.03

    hybrid = max(0.0, min(raw_hybrid, 0.9999))
    return round(hybrid, 6)


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


def retrieve_context(query: str, limit: int = 5):
    query_vector = get_embedding(query)

    initial_hits = search(query_vector=query_vector, limit=max(limit * 4, 12))
    initial_hits = _dedupe_hits(initial_hits)

    intent = _detect_query_intent(query)
    query_terms = _extract_query_terms(query)

    reranked = []
    for hit in initial_hits:
        hit_copy = dict(hit)
        hit_copy["vector_score"] = _normalize_vector_score(hit.get("score", 0.0))
        hit_copy["hybrid_score"] = _hybrid_score(hit_copy, query, query_terms, intent)
        hit_copy["intent"] = intent
        reranked.append(hit_copy)

    reranked.sort(
        key=lambda x: (
            x.get("hybrid_score", 0.0),
            x.get("vector_score", 0.0),
            -len((x.get("text", "") or "").split())
        ),
        reverse=True
    )

    final_hits = []
    for hit in reranked[:limit]:
        item = dict(hit)
        item["score"] = item["hybrid_score"]
        final_hits.append(item)

    return final_hits
