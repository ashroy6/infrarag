from __future__ import annotations

import re
from typing import Any

DEFAULT_CANDIDATE_TOP_K = 40
DEFAULT_FINAL_TOP_K = 6

QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
PAGE_RE = re.compile(r"\bpages?\s+\d+\s*(?:-|to)?\s*\d*\b", re.IGNORECASE)

SECTION_REFERENCE_RE = re.compile(
    r"\b(?:chapter|section|part|book)\s+\d{1,4}\b|\b[A-Z][A-Za-z0-9_-]{2,40}\s+\d{1,4}\b",
    re.IGNORECASE,
)

COMPARISON_MARKERS = (
    "compare",
    "difference between",
    "differences between",
    "versus",
    " vs ",
    "better than",
    "pros and cons",
    "advantages and disadvantages",
    "tradeoff",
    "trade-offs",
    "which is better",
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
    "connection refused",
    "permission denied",
    "access denied",
    "crash",
    "crashed",
    "stuck",
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
    r"^\s*who\s+was\s+",
    r"^\s*what\s+is\s+",
    r"^\s*what\s+are\s+",
    r"^\s*where\s+is\s+",
    r"^\s*where\s+are\s+",
    r"^\s*tell\s+me\s+about\s+",
    r"^\s*define\s+",
    r"^\s*meaning\s+of\s+",
)

YES_NO_PATTERNS = (
    r"^\s*is\s+",
    r"^\s*are\s+",
    r"^\s*was\s+",
    r"^\s*were\s+",
    r"^\s*do\s+",
    r"^\s*does\s+",
    r"^\s*did\s+",
    r"^\s*can\s+",
    r"^\s*could\s+",
    r"^\s*should\s+",
    r"^\s*would\s+",
    r"^\s*will\s+",
    r"^\s*has\s+",
    r"^\s*have\s+",
    r"^\s*had\s+",
)

HOW_TO_PATTERNS = (
    r"^\s*how\s+to\s+",
    r"^\s*how\s+do\s+i\s+",
    r"^\s*how\s+do\s+we\s+",
    r"^\s*how\s+can\s+i\s+",
    r"^\s*how\s+can\s+we\s+",
    r"^\s*how\s+should\s+i\s+",
    r"^\s*how\s+should\s+we\s+",
    r"^\s*steps\s+to\s+",
    r"^\s*best\s+way\s+to\s+",
)

LIST_PATTERNS = (
    r"^\s*list\s+",
    r"^\s*show\s+",
    r"^\s*give\s+me\s+",
    r"^\s*name\s+",
    r"^\s*names\s+of\s+",
    r"^\s*what\s+are\s+the\s+names\s+of\s+",
)

SOURCE_NAVIGATION_MARKERS = (
    "where is it mentioned",
    "where mentioned",
    "where does it say",
    "which file",
    "which document",
    "which source",
    "which page",
    "citation for",
    "source for",
    "find where",
    "show where",
)

ENTITY_PREFIX_PATTERNS = (
    r"^\s*who\s+is\s+(.+?)(?:\?|$)",
    r"^\s*who\s+was\s+(.+?)(?:\?|$)",
    r"^\s*what\s+is\s+(.+?)(?:\?|$)",
    r"^\s*what\s+are\s+(.+?)(?:\?|$)",
    r"^\s*where\s+is\s+(.+?)(?:\?|$)",
    r"^\s*where\s+are\s+(.+?)(?:\?|$)",
    r"^\s*tell\s+me\s+about\s+(.+?)(?:\?|$)",
    r"^\s*define\s+(.+?)(?:\?|$)",
    r"^\s*meaning\s+of\s+(.+?)(?:\?|$)",
)

ENTITY_TRAILING_NOISE_RE = re.compile(
    r"\b(in|inside|from|within|for|of|on|at|file|document|source|page|chapter|section)\b.*$",
    re.IGNORECASE,
)

ENTITY_LEADING_NOISE_RE = re.compile(
    r"^\s*(the|a|an|this|that|these|those)\s+",
    re.IGNORECASE,
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


def _looks_like_section_reference(q: str) -> bool:
    clean = _clean_query(q)
    if not clean:
        return False

    if SECTION_REFERENCE_RE.search(clean):
        section_words = (
            "what happens",
            "what is in",
            "summarize",
            "summarise",
            "summary",
            "explain",
            "describe",
            "tell me about",
        )
        lower = clean.lower()
        return any(marker in lower for marker in section_words)

    return False


def _looks_like_entity_lookup(q: str) -> bool:
    clean = _lower(q)
    return any(re.search(pattern, clean) for pattern in ENTITY_LOOKUP_PATTERNS)


def _extract_primary_entity(query: str) -> str:
    """
    Extract the entity from simple lookup questions.

    Examples:
      "who is mira" -> "mira"
      "tell me about Terraform" -> "Terraform"
      "what is RDS?" -> "RDS"

    This deliberately avoids clever LLM rewriting.
    """
    clean = _clean_query(query)
    if not clean:
        return ""

    for pattern in ENTITY_PREFIX_PATTERNS:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if not match:
            continue

        entity = str(match.group(1) or "").strip(" ?.,:;\"'")
        entity = ENTITY_TRAILING_NOISE_RE.sub("", entity).strip(" ?.,:;\"'")
        entity = ENTITY_LEADING_NOISE_RE.sub("", entity).strip(" ?.,:;\"'")
        entity = _clean_query(entity)

        if entity and len(entity) <= 80:
            return entity

    return ""


def _entity_rewritten_queries(entity: str, original_query: str) -> list[str]:
    """
    Entity lookup must search the entity, not only the full natural-language question.
    Keep this small to avoid FTS noise.
    """
    clean_entity = _clean_query(entity)
    clean_original = _clean_query(original_query)

    queries: list[str] = []
    if clean_entity:
        queries.extend(
            [
                clean_entity,
                f'"{clean_entity}"',
                f"{clean_entity} is",
                f"{clean_entity} was",
                f"{clean_entity},",
            ]
        )

    if clean_original and clean_original not in queries:
        queries.append(clean_original)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:5]


def _looks_like_yes_no_relationship(q: str) -> bool:
    clean = _lower(q)
    if not any(re.search(pattern, clean) for pattern in YES_NO_PATTERNS):
        return False

    relationship_markers = (
        " part of ",
        " included in ",
        " related to ",
        " belong to ",
        " used in ",
        " used for ",
        " required for ",
        " needed for ",
        " depend on ",
        " depends on ",
        " connected to ",
        " same as ",
        " different from ",
        " a type of ",
        " an example of ",
        " responsible for ",
        " support ",
        " supports ",
        " include ",
        " includes ",
        " contain ",
        " contains ",
        " mean ",
        " means ",
    )

    padded = f" {clean} "
    return any(marker in padded for marker in relationship_markers) or clean.endswith("?")


def _looks_like_how_to(q: str) -> bool:
    clean = _lower(q)
    return any(re.search(pattern, clean) for pattern in HOW_TO_PATTERNS)


def _looks_like_list_or_examples(q: str) -> bool:
    clean = _lower(q)
    list_markers = (
        "list of",
        "names of",
        "examples of",
        "types of",
        "tools",
        "services",
        "components",
        "stages",
        "steps",
        "benefits",
    )

    return any(re.search(pattern, clean) for pattern in LIST_PATTERNS) or any(
        marker in clean for marker in list_markers
    )


def _looks_like_source_navigation(q: str) -> bool:
    clean = _lower(q)
    return any(marker in clean for marker in SOURCE_NAVIGATION_MARKERS)


def _clean_comparison_entity(value: str) -> str:
    item = _clean_query(str(value or "").strip(" ?.,:;\"'"))
    item = re.sub(r"\b(the|a|an)\b", " ", item, flags=re.IGNORECASE)
    item = re.sub(
        r"\b(according to|in|inside|from|within|for|of|on|at)\b.*$",
        " ",
        item,
        flags=re.IGNORECASE,
    )
    item = _clean_query(item.strip(" ?.,:;\"'"))
    return item


def _extract_comparison_entities(query: str) -> list[str]:
    text = _clean_query(query)

    patterns = [
        r"\bcompare\s+(.+?)\s+(?:and|with|vs|versus)\s+(.+?)(?:\?|$)",
        r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bdifference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bexplain\s+the\s+difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+vs\s+(.+?)(?:\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        entities = []
        for group in match.groups():
            item = _clean_comparison_entity(group)
            if item:
                entities.append(item)

        if len(entities) >= 2:
            return entities[:2]

    return []


def _apply_fast_mode(plan: dict[str, Any]) -> dict[str, Any]:
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

    if query_shape in {"entity_lookup", "definition", "yes_no_relationship", "source_navigation"}:
        fast.update(
            {
                "candidate_top_k": 25,
                "final_top_k": 4,
                "keyword_top_k": 20,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: FTS5-first lookup, no vector noise.",
            }
        )
        return fast

    if query_shape in {"section_summary", "section_reference"}:
        fast.update(
            {
                "candidate_top_k": 45,
                "final_top_k": 6,
                "keyword_top_k": 35,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": False,
                "planner_reason": str(fast.get("planner_reason", "")) + " Fast mode: section hybrid search with nearby chunks, no reranker.",
            }
        )
        return fast

    if query_shape in {"comparison", "troubleshooting", "code_explanation", "how_to_steps", "list_or_examples"}:
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

    if query_shape in {"entity_lookup", "definition", "yes_no_relationship", "source_navigation"}:
        direct.update(
            {
                "candidate_top_k": 12,
                "final_top_k": 3,
                "keyword_top_k": 10,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": False,
                "use_reranker": False,
                "planner_reason": str(direct.get("planner_reason", "")) + " Direct mode: FTS5 lookup snippets only, no Ollama.",
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


def _finish_speed(plan: dict[str, Any], retrieval_speed: str) -> dict[str, Any]:
    if retrieval_speed == "direct":
        return _apply_direct_mode(plan)
    if retrieval_speed == "fast":
        return _apply_fast_mode(plan)
    return plan


def build_adaptive_retrieval_plan(
    query: str,
    base_plan: dict[str, Any] | None = None,
    retrieval_speed: str = "normal",
) -> dict[str, Any]:
    base = dict(base_plan or {})
    clean_query = _clean_query(query)
    quoted = _quoted_phrases(clean_query)

    pipeline = str(
        base.get("pipeline_used")
        or base.get("pipeline")
        or base.get("intent")
        or "normal_qa"
    )

    question_type = str(base.get("question_type") or "")
    source_strategy = str(base.get("source_strategy") or "cluster_by_best_source")

    plan: dict[str, Any] = {
        "query_shape": question_type or "normal_qa",
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
        "primary_entity": "",
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
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_comparison(clean_query) or question_type == "comparison":
        entities = _extract_comparison_entities(clean_query)
        rewritten = [clean_query]
        if len(entities) >= 2:
            rewritten.extend([
                f"{entities[0]} {entities[1]}",
                entities[0],
                entities[1],
            ])
        elif entities:
            rewritten.extend(entities)
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
        return _finish_speed(plan, retrieval_speed)

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
        return _finish_speed(plan, retrieval_speed)

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
        return _finish_speed(plan, retrieval_speed)

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
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_how_to(clean_query) or question_type == "how_to_steps":
        plan.update(
            {
                "query_shape": "how_to_steps",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 50),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 35,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": source_strategy,
                "planner_reason": "How-to query detected, so retrieval uses hybrid search plus nearby supporting chunks.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_yes_no_relationship(clean_query) or question_type == "yes_no_relationship":
        plan.update(
            {
                "query_shape": "yes_no_relationship",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 50),
                "final_top_k": max(plan["final_top_k"], 6),
                "keyword_top_k": 35,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": "cluster_by_best_source",
                "planner_reason": "Yes/no relationship query detected, so keyword-first hybrid retrieval is safer than planner-only routing.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_source_navigation(clean_query) or question_type == "source_navigation":
        plan.update(
            {
                "query_shape": "source_navigation",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 50),
                "final_top_k": max(plan["final_top_k"], 6),
                "keyword_top_k": 40,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": source_strategy,
                "planner_reason": "Source navigation query detected, so retrieval prioritizes exact source/citation matches.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_list_or_examples(clean_query) or question_type == "list_or_examples":
        plan.update(
            {
                "query_shape": "list_or_examples",
                "retrieval_mode": "keyword_first_hybrid",
                "candidate_top_k": max(plan["candidate_top_k"], 50),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 40,
                "neighbour_window": 0,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": "allow_multiple_sources",
                "planner_reason": "List/examples query detected, so retrieval allows multiple sources and prioritizes matching terms.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_section_reference(clean_query) or question_type in {"section_summary", "section_reference"}:
        plan.update(
            {
                "query_shape": "section_summary",
                "retrieval_mode": "section_retrieval",
                "candidate_top_k": max(plan["candidate_top_k"], 70),
                "final_top_k": max(plan["final_top_k"], 10),
                "keyword_top_k": 60,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": "cluster_by_best_source",
                "planner_reason": "Section/chapter reference detected, so retrieval uses section-aware hybrid search with nearby chunks.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

    if _looks_like_entity_lookup(clean_query) or question_type in {"direct_factual", "definition", "entity_lookup"}:
        shape = "definition" if question_type == "definition" else "entity_lookup"
        entity = _extract_primary_entity(clean_query)
        rewritten_queries = _entity_rewritten_queries(entity, clean_query) if entity else [clean_query]

        plan.update(
            {
                "query_shape": shape,
                "retrieval_mode": "entity_lookup",
                "primary_entity": entity,
                "rewritten_queries": rewritten_queries,
                "candidate_top_k": max(plan["candidate_top_k"], 80),
                "final_top_k": max(plan["final_top_k"], 8),
                "keyword_top_k": 60,
                "neighbour_window": 1,
                "use_keyword_search": True,
                "use_vector_search": True,
                "use_reranker": True,
                "source_strategy": "entity_disambiguation",
                "planner_reason": "Entity lookup detected, so retrieval extracts the entity, searches exact mentions first, scores identity evidence, and avoids early single-source clustering.",
            }
        )
        return _finish_speed(plan, retrieval_speed)

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
        return _finish_speed(plan, retrieval_speed)

    return _finish_speed(plan, retrieval_speed)
