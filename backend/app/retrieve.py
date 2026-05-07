from __future__ import annotations

import re
from typing import Any

from app.graph_retrieval import expand_with_graph_context
from app.hybrid_retrieve import adaptive_hybrid_retrieve
from app.query_planner import plan_query
from app.retrieval_planner import build_adaptive_retrieval_plan
from app.source_profile import get_source_profile

_PAGE_RANGE_RE = re.compile(
    r"\bpages?\s+(\d+)\s*(?:-|to)\s*(\d+)\b|\bpage\s+(\d+)\b",
    re.IGNORECASE,
)


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
    use_graph_context: bool = False,
    graph_max_chunks: int = 3,
    allowed_source_ids: list[str] | None = None,
    retrieval_speed: str = "normal",
) -> list[dict[str, Any]]:
    """
    Main retrieval entry point.

    Public signature stays the same so existing orchestrator, agents,
    and pipelines do not break.

    New behaviour:
    - router.py still decides answer pipeline
    - retrieval_planner.py decides retrieval mode/query shape
    - hybrid_retrieve.py performs vector + keyword + neighbour expansion
    """
    safe_limit = max(1, min(int(limit), 50))

    inferred_page_start, inferred_page_end = _extract_page_range(query)
    if page_start is None:
        page_start = inferred_page_start
    if page_end is None:
        page_end = inferred_page_end

    base_plan = retrieval_plan or plan_query(query)

    clean_retrieval_speed = "direct" if retrieval_speed == "direct" else ("fast" if retrieval_speed == "fast" else "normal")

    adaptive_plan = build_adaptive_retrieval_plan(
        query=query,
        base_plan=base_plan,
        retrieval_speed=clean_retrieval_speed,
    )

    source_profile = get_source_profile(source_id)
    adaptive_plan["source_profile"] = source_profile

    # Large structured/code sources benefit from neighbour expansion.
    if source_profile.get("source_size") == "large" and adaptive_plan.get("neighbour_window", 0) < 1:
        adaptive_plan["neighbour_window"] = 1

    final_hits = adaptive_hybrid_retrieve(
        query=query,
        retrieval_plan=adaptive_plan,
        limit=safe_limit,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
        allowed_source_ids=allowed_source_ids,
    )

    if use_graph_context and final_hits:
        final_hits, graph_meta = expand_with_graph_context(
            final_hits,
            max_graph_chunks=graph_max_chunks,
        )

        for hit in final_hits:
            hit.setdefault("graph_context_enabled", True)

        if final_hits:
            final_hits[0]["graph_context_meta"] = graph_meta

    return final_hits
