from __future__ import annotations

import re
from typing import Any

DEFAULT_CANDIDATE_TOP_K = 40
DEFAULT_FINAL_TOP_K = 6

QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
PAGE_RE = re.compile(r"\bpages?\s+\d+\s*(?:-|to)?\s*\d*\b", re.IGNORECASE)

COMPARISON_MARKERS = (
    "compare",
    "difference between",
    "differences between",
    "versus",
    " vs ",
    "better than",
    "pros and cons",
)

TROUBLESHOOTING_MARKERS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "timeout",
    "502",
    "500",
    "404",
    "not working",
    "broken",
    "debug",
    "troubleshoot",
    "why is this happening",
)

OVERVIEW_MARKERS = (
    "what is this document about",
    "what is this file about",
    "overview",
    "summarize this",
    "summarise this",
    "summary of this",
    "explain this document",
)

CODE_MARKERS = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".tf",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    "dockerfile",
    "function",
    "class",
    "module",
    "repo",
    "code",
)

ENTITY_LOOKUP_PATTERNS = (
    r"^\s*who\s+is\s+",
    r"^\s*what\s+is\s+",
    r"^\s*where\s+is\s+",
    r"^\s*tell\s+me\s+about\s+",
    r"^\s*define\s+",
)


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _lower(value: str) -> str:
    return _clean_query(value).lower()


def _quoted_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for match in QUOTE_RE.finditer(query or ""):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def _looks_like_comparison(q: str) -> bool:
    clean = f" {_lower(q)} "
    return any(marker in clean for marker in COMPARISON_MARKERS)


def _looks_like_troubleshooting(q: str) -> bool:
    clean = _lower(q)
    return any(marker in clean for marker in TROUBLESHOOTING_MARKERS)


def _looks_like_overview(q: str) -> bool:
    clean = _lower(q)
    return any(marker in clean for marker in OVERVIEW_MARKERS)


def _looks_like_code(q: str) -> bool:
    clean = _lower(q)
    return any(marker in clean for marker in CODE_MARKERS)


def _looks_like_entity_lookup(q: str) -> bool:
    clean = _lower(q)
    return any(re.search(pattern, clean) for pattern in ENTITY_LOOKUP_PATTERNS)


def _extract_comparison_entities(query: str) -> list[str]:
    text = _clean_query(query)

    patterns = [
        r"\bcompare\s+(.+?)\s+(?:and|with|vs|versus)\s+(.+?)(?:\?|$)",
        r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+vs\s+(.+?)(?:\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        entities = []
        for group in match.groups():
            item = re.sub(r"\b(the|a|an)\b", " ", group, flags=re.IGNORECASE)
            item = _clean_query(item.strip(" ?.,:;"))
            if item:
                entities.append(item)
        if len(entities) >= 2:
            return entities[:2]

    return []


def _apply_fast_mode(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Fast mode trades depth for speed.

    It is intended for quick factual/entity/exact searches, not deep summaries.
    """
    fast = dict(plan)
    fast["retrieval_speed"] = "fast"

    query_shape = str(fast.get("query_shape") or "normal_qa")

    if query_shape == "exact_phrase":
        fast.update(
            {
                "candidate_top_k": 10,
                "final_top_k": 3,
                "keyword_top_k": 8,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: FTS5 only, top 3 chunks, no reranker.",
            }
        )
        return fast

    if query_shape == "entity_lookup":
        fast.update(
            {
                "candidate_top_k": 15,
                "final_top_k": 3,
                "keyword_top_k": 10,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: FTS5-first entity lookup, top 3 chunks, no reranker.",
            }
        )
        return fast

    if query_shape in {"comparison", "troubleshooting", "code_explanation"}:
        fast.update(
            {
                "candidate_top_k": 25,
                "final_top_k": 4,
                "keyword_top_k": 15,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": False,
                "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: smaller hybrid search, no reranker.",
            }
        )
        return fast

    fast.update(
        {
            "candidate_top_k": 20,
            "final_top_k": 3,
            "keyword_top_k": 10,
            "neighbour_window": 0,
            "use_reranker": False,
            "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: reduced search breadth and shorter context.",
        }
    )
    return fast


def _apply_direct_mode(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Direct mode is fastest.

    It retrieves only the strongest evidence and lets the backend return
    a deterministic citation/snippet answer without calling Ollama for
    simple lookup/search questions.
    """
    direct = dict(plan)
    direct["retrieval_speed"] = "direct"

    query_shape = str(direct.get("query_shape") or "normal_qa")

    if query_shape == "exact_phrase":
        direct.update(
            {
                "candidate_top_k": 6,
                "final_top_k": 2,
                "keyword_top_k": 5,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(direct.get("planner_reason", "")) + " Direct mode: exact FTS5 snippets only, no Ollama.",
            }
        )
        return direct

    if query_shape == "entity_lookup":
        direct.update(
            {
                "candidate_top_k": 8,
                "final_top_k": 3,
                "keyword_top_k": 6,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(direct.get("planner_reason", "")) + " Direct mode: FTS5 entity snippets only, no Ollama.",
            }
        )
        return direct

    direct.update(
        {
            "candidate_top_k": 10,
            "final_top_k": 3,
            "keyword_top_k": 8,
            "neighbour_window": 0,
            "use_keyword_search": True,
            "use_vector_search": False,
            "use_reranker": False,
            "planner_reason": str(direct.get("planner_reason", "")) + " Direct mode: reduced FTS5-only lookup.",
        }
    )
    return direct


def build_adaptive_retrieval_plan(
    query: str,
    base_plan: dict[str, Any] | None = None,
    retrieval_speed: str = "normal",
) -> dict[str, Any]:
    """
    Deterministic retrieval planner.

    This does not decide the answer pipeline.
    router.py still decides pipeline.
    This only decides how retrieval should search.
    """
    base = dict(base_plan or {})
    clean_query = _clean_query(query)
    quoted = _quoted_phrases(clean_query)

    pipeline = str(
        base.get("pipeline_used")
        or base.get("pipeline")
        or base.get("intent")
        or "normal_qa"
    )

    source_strategy = str(base.get("source_strategy") or "cluster_by_best_source")

    plan: dict[str, Any] = {
        "query_shape": "normal_qa",
        "retrieval_mode": "vector_rerank",
        "rewritten_queries": base.get("rewritten_queries") or [clean_query],
        "candidate_top_k": int(base.get("candidate_top_k", DEFAULT_CANDIDATE_TOP_K) or DEFAULT_CANDIDATE_TOP_K),
        "final_top_k": int(base.get("final_top_k", DEFAULT_FINAL_TOP_K) or DEFAULT_FINAL_TOP_K),
        "source_strategy": source_strategy,
        "keyword_top_k": 30,
        "neighbour_window": 0,
        "use_keyword_search": False,
        "use_vector_search": True,
        "use_reranker": True,
        "comparison_entities": [],
        "exact_phrases": quoted,
        "planner_reason": "Default vector + rerank retrieval.",
        "retrieval_speed": "normal",
    }

    if quoted:
        plan.update(
            {
                "query_shape": "exact_phrase",
                "retrieval_mode": "exact_phrase_search",
                "candidate_top_k": max(plan["candidate_top_k"], 60),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 50,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "source_strategy": "cluster_by_best_source",
                "planner_reason": "Quoted phrase detected, so exact phrase retrieval uses FTS5 only and disables vector/reranker noise.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if _looks_like_comparison(clean_query):
        entities = _extract_comparison_entities(clean_query)
        rewritten = [clean_query] + entities if entities else [clean_query]
        plan.update(
            {
                "query_shape": "comparison",
                "retrieval_mode": "balanced_multi_entity_hybrid",
                "rewritten_queries": rewritten[:5],
                "candidate_top_k": max(plan["candidate_top_k"], 70),
                "final_top_k": max(plan["final_top_k"], 10),
                "keyword_top_k": 50,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "source_strategy": "allow_multiple_sources",
                "comparison_entities": entities,
                "planner_reason": "Comparison query detected, so retrieval must preserve evidence from multiple entities/sources.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if _looks_like_troubleshooting(clean_query) or pipeline == "incident_runbook":
        plan.update(
            {
                "query_shape": "troubleshooting",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 60),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 40,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "source_strategy": "allow_multiple_sources",
                "planner_reason": "Troubleshooting query detected, so exact error terms and nearby chunks matter.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if _looks_like_overview(clean_query) or pipeline == "document_summary":
        plan.update(
            {
                "query_shape": "overview",
                "retrieval_mode": "section_overview",
                "candidate_top_k": max(plan["candidate_top_k"], 80),
                "final_top_k": max(plan["final_top_k"], 12),
                "keyword_top_k": 20,
                "neighbour_window": 2,
                "use_keyword_search": True,
                "source_strategy": "cluster_by_best_source",
                "planner_reason": "Overview query detected, so section-leading and neighbouring chunks are useful.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if _looks_like_code(clean_query) or pipeline == "repo_explanation":
        plan.update(
            {
                "query_shape": "code_explanation",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 70),
                "final_top_k": max(plan["final_top_k"], 10),
                "keyword_top_k": 40,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "source_strategy": source_strategy,
                "planner_reason": "Code/repo query detected, so filenames, symbols, and adjacent chunks matter.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if _looks_like_entity_lookup(clean_query):
        plan.update(
            {
                "query_shape": "entity_lookup",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 60),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 40,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "source_strategy": "cluster_by_best_source",
                "planner_reason": "Entity lookup detected, so keyword-first hybrid retrieval is safer than vector-only retrieval.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    if PAGE_RE.search(clean_query):
        plan.update(
            {
                "query_shape": "section_summary",
                "retrieval_mode": "section_topic_retrieval",
                "candidate_top_k": max(plan["candidate_top_k"], 50),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 20,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "planner_reason": "Page/section reference detected, so neighbouring chunks are useful.",
            }
        )
        return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)

    return _apply_direct_mode(plan) if retrieval_speed == "direct" else (_apply_fast_mode(plan) if retrieval_speed == "fast" else plan)
