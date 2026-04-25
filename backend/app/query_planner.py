from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from app.llm_client import generate_text
from app.prompts import QUERY_PLANNER_PROMPT

PLANNER_ENABLED = os.getenv("PLANNER_ENABLED", "true").lower() == "true"
PLANNER_TIMEOUT_SECONDS = int(os.getenv("PLANNER_TIMEOUT_SECONDS", "45"))
PLANNER_NUM_PREDICT = int(os.getenv("PLANNER_NUM_PREDICT", "220"))

VALID_PIPELINES = {
    "normal_qa",
    "long_explanation",
    "document_summary",
    "repo_explanation",
    "incident_runbook",
}

PIPELINE_LABELS = {
    "normal_qa": "Normal RAG Q&A",
    "long_explanation": "Long explanation",
    "document_summary": "Full document/book summarisation",
    "repo_explanation": "Terraform/repo explanation",
    "incident_runbook": "Incident/runbook troubleshooting",
}

VALID_SOURCE_STRATEGIES = {
    "cluster_by_best_source",
    "allow_multiple_sources",
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())



def _looks_like_comparison_question(question: str) -> bool:
    q = _clean_text(question).lower()
    comparison_markers = [
        "compare",
        "contrast",
        "difference",
        "differentiate",
        "versus",
        " vs ",
        "both",
        "each",
        "between",
        "what problem does each",
        "what do they prevent",
    ]
    return any(marker in q for marker in comparison_markers)


def _fallback_plan(question: str, reason: str) -> dict[str, Any]:
    cleaned = _clean_text(question)

    return {
        "pipeline": "normal_qa",
        "pipeline_used": "normal_qa",
        "pipeline_label": PIPELINE_LABELS["normal_qa"],
        "intent": "normal_qa",
        "question_type": "fallback",
        "rewritten_queries": [cleaned] if cleaned else [],
        "needs_full_document": False,
        "candidate_top_k": 40,
        "final_top_k": 6,
        "source_strategy": "cluster_by_best_source",
        "answer_style": "concise",
        "confidence": 0.5,
        "reason": reason,
        "router": "planner_fallback",
    }


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()

    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clean_queries(question: str, queries: Any) -> list[str]:
    original = _clean_text(question)
    out: list[str] = []

    if isinstance(queries, list):
        for item in queries:
            value = _clean_text(str(item or ""))
            if value and value not in out:
                out.append(value)

    if original and original not in out:
        out.insert(0, original)

    return out[:3]


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    return max(minimum, min(number, maximum))


def _clamp_float(value: Any, default: float = 0.6) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    return max(0.0, min(number, 1.0))


def _normalize_plan(question: str, data: dict[str, Any], router_name: str) -> dict[str, Any]:
    pipeline = str(data.get("pipeline") or data.get("intent") or "normal_qa").strip()

    if pipeline not in VALID_PIPELINES:
        pipeline = "normal_qa"

    source_strategy = str(data.get("source_strategy") or "cluster_by_best_source").strip()
    if source_strategy not in VALID_SOURCE_STRATEGIES:
        source_strategy = "cluster_by_best_source"

    answer_style = str(data.get("answer_style") or "concise").strip().lower()
    if answer_style not in {"concise", "balanced", "detailed"}:
        answer_style = "concise"

    if pipeline == "long_explanation":
        answer_style = "detailed"

    if pipeline == "document_summary":
        answer_style = "detailed"

    if pipeline == "incident_runbook":
        answer_style = "balanced"

    final_top_k_default = {
        "normal_qa": 6,
        "long_explanation": 8,
        "document_summary": 10,
        "repo_explanation": 8,
        "incident_runbook": 8,
    }.get(pipeline, 6)

    candidate_top_k_default = {
        "normal_qa": 40,
        "long_explanation": 50,
        "document_summary": 60,
        "repo_explanation": 50,
        "incident_runbook": 50,
    }.get(pipeline, 40)

    if _looks_like_comparison_question(question):
        if pipeline == "repo_explanation":
            pipeline = "long_explanation"
            answer_style = "detailed"
        source_strategy = "allow_multiple_sources"

    return {
        "pipeline": pipeline,
        "pipeline_used": pipeline,
        "pipeline_label": PIPELINE_LABELS[pipeline],
        "intent": pipeline,
        "question_type": str(data.get("question_type") or pipeline)[:80],
        "rewritten_queries": _clean_queries(question, data.get("rewritten_queries")),
        "needs_full_document": bool(data.get("needs_full_document", pipeline == "document_summary")),
        "candidate_top_k": _clamp_int(data.get("candidate_top_k"), candidate_top_k_default, 20, 80),
        "final_top_k": _clamp_int(data.get("final_top_k"), final_top_k_default, 3, 12),
        "source_strategy": source_strategy,
        "answer_style": answer_style,
        "confidence": _clamp_float(data.get("confidence"), 0.6),
        "reason": str(data.get("reason") or "Planned by local LLM.")[:300],
        "router": router_name,
    }


@lru_cache(maxsize=256)
def _plan_cached(question: str, chat_context: str) -> str:
    prompt = QUERY_PLANNER_PROMPT.format(
        question=question,
        chat_context=chat_context or "No recent conversation context.",
    )

    return generate_text(
        prompt,
        temperature=0.0,
        num_predict=PLANNER_NUM_PREDICT,
        timeout=PLANNER_TIMEOUT_SECONDS,
    )


def plan_query(question: str, chat_context: str = "") -> dict[str, Any]:
    cleaned_question = _clean_text(question)

    if not cleaned_question:
        return _fallback_plan(question, "Empty question.")

    if not PLANNER_ENABLED:
        return _fallback_plan(question, "Planner disabled.")

    try:
        raw = _plan_cached(cleaned_question, chat_context or "")
        data = _extract_json(raw)

        if not data:
            return _fallback_plan(question, "Planner returned invalid JSON.")

        return _normalize_plan(cleaned_question, data, "llm_planner")

    except Exception as exc:
        return _fallback_plan(question, f"Planner failed: {exc}")
