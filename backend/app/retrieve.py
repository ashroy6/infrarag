from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.embedding_service import get_embedding
from app.qdrant_client import search
from app.query_planner import plan_query
from app.reranker import rerank_hits

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "about", "into", "over", "under", "than", "how", "why", "when", "where",
    "can", "could", "should", "would", "will", "may", "might", "tell", "me",
    "explain", "give", "show", "summarise", "summarize"
}

_PAGE_RANGE_RE = re.compile(
    r"\bpages?\s+(\d+)\s*(?:-|to)\s*(\d+)\b|\bpage\s+(\d+)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-/.]+", normalized)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


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


def _source_key(hit: dict[str, Any]) -> str:
    return str(hit.get("source_id") or hit.get("source") or "unknown")


def _query_key(hit: dict[str, Any]) -> int:
    try:
        return int(hit.get("retrieval_query_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for hit in hits:
        key = (
            hit.get("source_id") or hit.get("source", ""),
            hit.get("chunk_index", -1),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(hit)

    return deduped


def _keyword_overlap_score(query: str, text: str, source: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0

    haystack = set(_tokenize(f"{source}\n{text}"))
    matched = query_terms.intersection(haystack)

    if not matched:
        return 0.0

    return min(1.0, len(matched) / max(len(query_terms), 1))


def _normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(value, 1.0))


def _hit_score(hit: dict[str, Any]) -> float:
    for key in ("reranker_score", "score", "pre_cluster_score", "vector_score"):
        if key in hit:
            return _normalize_score(hit.get(key))
    return 0.0


def _source_cluster_scores(hits: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        groups[_source_key(hit)].append(hit)

    scores_by_source: dict[str, float] = {}

    for key, items in groups.items():
        scores = sorted(
            [_normalize_score(item.get("pre_cluster_score", item.get("score", 0.0))) for item in items],
            reverse=True,
        )

        top_scores = scores[:5]
        avg_top = sum(top_scores) / max(len(top_scores), 1)
        max_score = max(scores) if scores else 0.0
        count_bonus = min(len(items), 6) * 0.025

        scores_by_source[key] = round((0.65 * avg_top) + (0.25 * max_score) + count_bonus, 6)

    return scores_by_source


def _best_source_per_query(hits: list[dict[str, Any]]) -> set[str]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[_query_key(hit)].append(hit)

    selected: set[str] = set()

    for _, query_hits in grouped.items():
        source_scores = _source_cluster_scores(query_hits)
        if not source_scores:
            continue

        best_source = max(source_scores.items(), key=lambda item: item[1])[0]
        selected.add(best_source)

    return selected


def _preserve_query_diversity(hits: list[dict[str, Any]], per_query: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[_query_key(hit)].append(hit)

    preserved: list[dict[str, Any]] = []

    for _, query_hits in grouped.items():
        query_hits.sort(
            key=lambda item: (
                _normalize_score(item.get("pre_cluster_score", item.get("score", 0.0))),
                _normalize_score(item.get("vector_score", 0.0)),
            ),
            reverse=True,
        )
        preserved.extend(query_hits[:per_query])

    return _dedupe_hits(preserved)


def _cluster_sources(hits: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    if not hits:
        return []

    source_scores = _source_cluster_scores(hits)

    if not source_scores:
        return []

    ranked_sources = sorted(source_scores.items(), key=lambda item: item[1], reverse=True)

    if strategy == "allow_multiple_sources":
        selected_keys = set()

        # Keep the strongest source per rewritten query so one topic cannot dominate.
        selected_keys.update(_best_source_per_query(hits))

        # Also keep a few globally strong source clusters.
        selected_keys.update(key for key, _ in ranked_sources[:5])
    else:
        selected_keys = {ranked_sources[0][0]}

    clustered: list[dict[str, Any]] = []

    for hit in hits:
        key = _source_key(hit)
        if key not in selected_keys:
            continue

        item = dict(hit)
        cluster_score = source_scores.get(key, 0.0)
        item["source_cluster_score"] = cluster_score
        item["score"] = round(
            (0.78 * _normalize_score(item.get("score", 0.0))) + (0.22 * cluster_score),
            6,
        )
        clustered.append(item)

    clustered.sort(
        key=lambda item: (
            item.get("score", 0.0),
            item.get("source_cluster_score", 0.0),
            item.get("vector_score", 0.0),
        ),
        reverse=True,
    )

    return clustered


def _diversify_final_hits(hits: list[dict[str, Any]], top_n: int, strategy: str) -> list[dict[str, Any]]:
    if strategy != "allow_multiple_sources" or not hits:
        return hits[:top_n]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[_source_key(hit)].append(hit)

    for source_hits in grouped.values():
        source_hits.sort(key=_hit_score, reverse=True)

    ranked_sources = sorted(
        grouped.keys(),
        key=lambda source_id: _hit_score(grouped[source_id][0]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen = set()

    # First pass: keep best hit from several sources.
    for source_id in ranked_sources[: min(4, top_n)]:
        hit = grouped[source_id][0]
        key = (_source_key(hit), hit.get("chunk_index", -1))
        if key not in seen:
            seen.add(key)
            selected.append(hit)

    # Second pass: fill remaining slots by best overall score.
    for hit in sorted(hits, key=_hit_score, reverse=True):
        if len(selected) >= top_n:
            break

        key = (_source_key(hit), hit.get("chunk_index", -1))
        if key in seen:
            continue

        seen.add(key)
        selected.append(hit)

    selected.sort(key=_hit_score, reverse=True)
    return selected[:top_n]


def _multi_query_search(
    queries: list[str],
    candidate_top_k: int,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
) -> list[dict[str, Any]]:
    all_hits: list[dict[str, Any]] = []

    for query_index, retrieval_query in enumerate(queries):
        query_vector = get_embedding(retrieval_query)

        hits = search(
            query_vector=query_vector,
            limit=candidate_top_k,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
        )

        for hit in hits:
            item = dict(hit)
            vector_score = _normalize_score(hit.get("score", 0.0))
            lexical_score = _keyword_overlap_score(
                retrieval_query,
                hit.get("text", "") or "",
                hit.get("source", "") or "",
            )

            item["vector_score"] = vector_score
            item["lexical_score"] = round(lexical_score, 6)
            item["retrieval_query"] = retrieval_query
            item["retrieval_query_index"] = query_index
            item["pre_cluster_score"] = round((0.76 * vector_score) + (0.24 * lexical_score), 6)
            item["score"] = item["pre_cluster_score"]

            all_hits.append(item)

    return _dedupe_hits(all_hits)


def retrieve_context(
    query: str,
    limit: int = 5,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    retrieval_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))

    inferred_page_start, inferred_page_end = _extract_page_range(query)
    if page_start is None:
        page_start = inferred_page_start
    if page_end is None:
        page_end = inferred_page_end

    plan = retrieval_plan or plan_query(query)

    rewritten_queries = plan.get("rewritten_queries") or [query]
    rewritten_queries = [
        re.sub(r"\s+", " ", str(item or "").strip())
        for item in rewritten_queries
        if str(item or "").strip()
    ]

    if query not in rewritten_queries:
        rewritten_queries.insert(0, query)

    rewritten_queries = rewritten_queries[:5]

    candidate_top_k = max(
        safe_limit * 5,
        int(plan.get("candidate_top_k", 40) or 40),
    )
    candidate_top_k = max(10, min(candidate_top_k, 80))

    source_strategy = str(plan.get("source_strategy") or "cluster_by_best_source")

    raw_hits = _multi_query_search(
        queries=rewritten_queries,
        candidate_top_k=candidate_top_k,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    if not raw_hits:
        return []

    if source_strategy == "allow_multiple_sources":
        raw_hits = _preserve_query_diversity(
            raw_hits,
            per_query=max(8, safe_limit),
        )

    clustered_hits = _cluster_sources(raw_hits, source_strategy)

    if not clustered_hits:
        return []

    final_top_k = max(safe_limit, int(plan.get("final_top_k", safe_limit) or safe_limit))
    final_top_k = max(1, min(final_top_k, 12))

    reranked = rerank_hits(
        question=query,
        hits=clustered_hits,
        top_n=max(final_top_k * 2, final_top_k),
    )

    diversified = _diversify_final_hits(
        reranked,
        top_n=final_top_k,
        strategy=source_strategy,
    )

    return diversified[:safe_limit]
