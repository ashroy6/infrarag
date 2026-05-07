from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from app.embedding_service import get_embedding
from app.keyword_index import search_keyword_index
from app.metadata_db import MetadataDB
from app.qdrant_client import get_chunks_by_refs, get_chunks_by_source_id, search
from app.reranker import rerank_hits

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "about", "into", "over", "under", "than", "how", "why", "when", "where",
    "can", "could", "should", "would", "will", "may", "might", "tell", "me",
    "explain", "give", "show", "summarise", "summarize", "please"
}

MAX_KEYWORD_SCAN_SOURCES = int(os.getenv("MAX_KEYWORD_SCAN_SOURCES", "80"))
MAX_KEYWORD_SCAN_CHUNKS_PER_SOURCE = int(os.getenv("MAX_KEYWORD_SCAN_CHUNKS_PER_SOURCE", "3000"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-/.]+", normalized)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, 1.0))


def source_key(hit: dict[str, Any]) -> str:
    return str(hit.get("source_id") or hit.get("source") or "unknown")


def hit_key(hit: dict[str, Any]) -> tuple[str, int]:
    try:
        chunk_index = int(hit.get("chunk_index", -1))
    except (TypeError, ValueError):
        chunk_index = -1
    return source_key(hit), chunk_index


def dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []

    for hit in hits:
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)

    return deduped


def keyword_score(query: str, text: str, source: str = "", exact_phrases: list[str] | None = None) -> float:
    haystack_text = normalize_text(f"{source}\n{text}")
    query_terms = set(tokenize(query))
    haystack_terms = set(tokenize(haystack_text))

    if not query_terms:
        return 0.0

    matched = query_terms.intersection(haystack_terms)
    overlap = len(matched) / max(len(query_terms), 1)

    exact_bonus = 0.0
    for phrase in exact_phrases or []:
        clean_phrase = normalize_text(phrase)
        if clean_phrase and clean_phrase in haystack_text:
            exact_bonus = max(exact_bonus, 1.0)

    return round(min(1.0, (0.70 * overlap) + (0.30 * exact_bonus)), 6)


def multi_query_vector_search(
    *,
    queries: list[str],
    candidate_top_k: int,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
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
            allowed_source_ids=allowed_source_ids,
        )

        for hit in hits:
            item = dict(hit)
            vector_score = normalize_score(item.get("score", 0.0))
            lexical_score = keyword_score(retrieval_query, item.get("text", ""), item.get("source", ""))

            item["retrieval_query"] = retrieval_query
            item["retrieval_query_index"] = query_index
            item["retrieval_channel"] = "vector"
            item["vector_score"] = vector_score
            item["lexical_score"] = lexical_score
            item["pre_cluster_score"] = round((0.76 * vector_score) + (0.24 * lexical_score), 6)
            item["score"] = item["pre_cluster_score"]

            all_hits.append(item)

    return dedupe_hits(all_hits)


def _source_allowed(source_id: str, allowed_source_ids: list[str] | None) -> bool:
    if allowed_source_ids is None:
        return True
    clean = {str(item).strip() for item in allowed_source_ids if str(item).strip()}
    return source_id in clean


def _candidate_source_ids(
    *,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[str]:
    if source_id:
        return [source_id] if _source_allowed(source_id, allowed_source_ids) else []

    db = MetadataDB()
    records = db.list_active_files(
        source_type=source_type,
    )

    source_ids: list[str] = []
    for record in records:
        sid = str(record.get("source_id") or "").strip()
        if not sid:
            continue

        if allowed_source_ids is not None and not _source_allowed(sid, allowed_source_ids):
            continue

        if file_type and str(record.get("file_type") or "") != file_type:
            continue

        source_ids.append(sid)

        if len(source_ids) >= MAX_KEYWORD_SCAN_SOURCES:
            break

    return source_ids


def keyword_search(
    *,
    query: str,
    queries: list[str],
    keyword_top_k: int,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
    exact_phrases: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Fast keyword search using SQLite FTS5.

    Old behaviour scanned all chunks manually in Python.
    New behaviour asks SQLite FTS5 for matching chunks first.
    """
    scored: list[dict[str, Any]] = []

    # Search each rewritten query. This helps comparison/entity queries.
    for retrieval_query in queries:
        fts_hits = search_keyword_index(
            query=retrieval_query,
            limit=keyword_top_k,
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
            exact_phrases=exact_phrases,
        )

        for hit in fts_hits:
            text = hit.get("text", "") or ""
            source_path = hit.get("source", "") or ""

            lexical_score = keyword_score(
                retrieval_query,
                text,
                source_path,
                exact_phrases=exact_phrases,
            )

            # FTS found the row, so do not let lexical scoring collapse to zero.
            lexical_score = max(lexical_score, normalize_score(hit.get("score", 0.0)))

            item = dict(hit)
            item["retrieval_query"] = retrieval_query
            item["retrieval_channel"] = "fts_keyword"
            item["keyword_score"] = lexical_score
            item["lexical_score"] = lexical_score
            item["vector_score"] = normalize_score(item.get("vector_score", 0.0))
            item["pre_cluster_score"] = round((0.88 * lexical_score) + (0.12 * normalize_score(item.get("score", 0.0))), 6)
            item["score"] = item["pre_cluster_score"]

            scored.append(item)

    scored = dedupe_hits(scored)
    scored.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)

    return scored[:keyword_top_k]


def source_cluster_scores(hits: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[source_key(hit)].append(hit)

    scores: dict[str, float] = {}

    for sid, items in grouped.items():
        item_scores = sorted(
            [normalize_score(item.get("pre_cluster_score", item.get("score", 0.0))) for item in items],
            reverse=True,
        )
        top_scores = item_scores[:5]
        avg_top = sum(top_scores) / max(len(top_scores), 1)
        max_score = max(item_scores) if item_scores else 0.0
        count_bonus = min(len(items), 6) * 0.025
        keyword_bonus = 0.04 if any(item.get("retrieval_channel") == "keyword" for item in items) else 0.0

        scores[sid] = round((0.62 * avg_top) + (0.26 * max_score) + count_bonus + keyword_bonus, 6)

    return scores


def cluster_sources(hits: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    if not hits:
        return []

    scores = source_cluster_scores(hits)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked:
        return []

    if strategy == "allow_multiple_sources":
        selected_sources = {sid for sid, _ in ranked[:5]}
    else:
        selected_sources = {ranked[0][0]}

    clustered: list[dict[str, Any]] = []

    for hit in hits:
        sid = source_key(hit)
        if sid not in selected_sources:
            continue

        item = dict(hit)
        cluster_score = scores.get(sid, 0.0)
        item["source_cluster_score"] = cluster_score
        item["score"] = round((0.78 * normalize_score(item.get("score", 0.0))) + (0.22 * cluster_score), 6)
        clustered.append(item)

    clustered.sort(
        key=lambda item: (
            normalize_score(item.get("score", 0.0)),
            normalize_score(item.get("source_cluster_score", 0.0)),
            normalize_score(item.get("keyword_score", 0.0)),
            normalize_score(item.get("vector_score", 0.0)),
        ),
        reverse=True,
    )

    return dedupe_hits(clustered)


def diversify_final_hits(hits: list[dict[str, Any]], top_n: int, strategy: str) -> list[dict[str, Any]]:
    if strategy != "allow_multiple_sources" or not hits:
        return hits[:top_n]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[source_key(hit)].append(hit)

    for source_hits in grouped.values():
        source_hits.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)

    ranked_sources = sorted(
        grouped.keys(),
        key=lambda sid: normalize_score(grouped[sid][0].get("score", 0.0)),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for sid in ranked_sources[: min(4, top_n)]:
        hit = grouped[sid][0]
        key = hit_key(hit)
        if key not in seen:
            seen.add(key)
            selected.append(hit)

    for hit in sorted(hits, key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True):
        if len(selected) >= top_n:
            break
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)

    selected.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)
    return selected[:top_n]


def expand_neighbour_chunks(hits: list[dict[str, Any]], neighbour_window: int) -> list[dict[str, Any]]:
    if neighbour_window <= 0 or not hits:
        return hits

    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for hit in hits:
        sid = source_key(hit)
        try:
            idx = int(hit.get("chunk_index", -1))
        except (TypeError, ValueError):
            continue

        if not sid or idx < 0:
            continue

        for n in range(idx - neighbour_window, idx + neighbour_window + 1):
            if n < 0:
                continue
            key = (sid, n)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"source_id": sid, "chunk_index": n})

    neighbours = get_chunks_by_refs(refs)

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for hit in hits:
        by_key[hit_key(hit)] = hit

    for chunk in neighbours:
        key = hit_key(chunk)
        if key in by_key:
            continue

        item = dict(chunk)
        item["retrieval_channel"] = "neighbour"
        item["score"] = max(0.01, normalize_score(item.get("score", 0.0)) * 0.85)
        item["neighbour_added"] = True
        by_key[key] = item

    merged = list(by_key.values())
    merged.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)
    return merged


def adaptive_hybrid_retrieve(
    *,
    query: str,
    retrieval_plan: dict[str, Any],
    limit: int,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    allowed_source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))

    queries = retrieval_plan.get("rewritten_queries") or [query]
    queries = [
        re.sub(r"\s+", " ", str(item or "").strip())
        for item in queries
        if str(item or "").strip()
    ]
    if query not in queries:
        queries.insert(0, query)
    queries = queries[:5]

    candidate_top_k = max(safe_limit * 5, int(retrieval_plan.get("candidate_top_k", 40) or 40))
    candidate_top_k = max(10, min(candidate_top_k, 100))

    final_top_k = max(safe_limit, int(retrieval_plan.get("final_top_k", safe_limit) or safe_limit))
    final_top_k = max(1, min(final_top_k, 16))

    keyword_top_k = max(final_top_k, int(retrieval_plan.get("keyword_top_k", 30) or 30))
    keyword_top_k = max(5, min(keyword_top_k, 80))

    source_strategy = str(retrieval_plan.get("source_strategy") or "cluster_by_best_source")
    use_vector = bool(retrieval_plan.get("use_vector_search", True))
    use_keyword = bool(retrieval_plan.get("use_keyword_search", False))
    use_reranker = bool(retrieval_plan.get("use_reranker", True))

    exact_phrases = retrieval_plan.get("exact_phrases") or []

    all_hits: list[dict[str, Any]] = []

    if use_vector:
        all_hits.extend(
            multi_query_vector_search(
                queries=queries,
                candidate_top_k=candidate_top_k,
                source_id=source_id,
                source=source,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
                allowed_source_ids=allowed_source_ids,
            )
        )

    if use_keyword:
        all_hits.extend(
            keyword_search(
                query=query,
                queries=queries,
                keyword_top_k=keyword_top_k,
                source_id=source_id,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
                allowed_source_ids=allowed_source_ids,
                exact_phrases=exact_phrases,
            )
        )

    all_hits = dedupe_hits(all_hits)
    if not all_hits:
        return []

    clustered = cluster_sources(all_hits, source_strategy)
    if not clustered:
        return []

    neighbour_window = int(retrieval_plan.get("neighbour_window", 0) or 0)
    if neighbour_window > 0:
        clustered = expand_neighbour_chunks(clustered[:final_top_k], neighbour_window=neighbour_window)

    if use_reranker:
        reranked = rerank_hits(
            question=query,
            hits=clustered,
            top_n=max(final_top_k * 2, final_top_k),
        )
    else:
        reranked = clustered[: max(final_top_k * 2, final_top_k)]

    diversified = diversify_final_hits(
        reranked,
        top_n=final_top_k,
        strategy=source_strategy,
    )

    final_hits = diversified[:safe_limit]

    if use_keyword and use_vector:
        retriever_used = "FTS5 + Qdrant"
    elif use_keyword:
        retriever_used = "FTS5"
    elif use_vector:
        retriever_used = "Qdrant"
    else:
        retriever_used = "none"

    for hit in final_hits:
        hit.setdefault("adaptive_retrieval", True)
        hit.setdefault("retrieval_mode", retrieval_plan.get("retrieval_mode", "vector_rerank"))
        hit.setdefault("retrieval_speed", retrieval_plan.get("retrieval_speed", "normal"))
        hit.setdefault("retriever_used", retriever_used)
        hit.setdefault("query_shape", retrieval_plan.get("query_shape", "normal_qa"))
        hit.setdefault("reranker_used", bool(use_reranker))
        hit.setdefault("neighbour_window", int(retrieval_plan.get("neighbour_window", 0) or 0))
        hit.setdefault("retrieval_planner_reason", retrieval_plan.get("planner_reason", ""))

    return final_hits
