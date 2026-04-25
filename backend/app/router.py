from __future__ import annotations

from typing import Any

from app.query_planner import PIPELINE_LABELS, VALID_PIPELINES, plan_query


def decide_intent(question: str, chat_context: str = "") -> dict[str, Any]:
    plan = plan_query(question, chat_context=chat_context)

    pipeline = plan.get("pipeline_used") or plan.get("pipeline") or "normal_qa"
    if pipeline not in VALID_PIPELINES:
        pipeline = "normal_qa"

    return {
        "intent": pipeline,
        "pipeline_used": pipeline,
        "pipeline_label": PIPELINE_LABELS[pipeline],
        "answer_length": plan.get("answer_style", "balanced"),
        "needs_all_chunks": bool(plan.get("needs_full_document", False)),
        "confidence": float(plan.get("confidence", 0.5)),
        "reason": str(plan.get("reason", "Planned by local LLM."))[:300],
        "router": plan.get("router", "llm_planner"),
        "question_type": plan.get("question_type"),
        "rewritten_queries": plan.get("rewritten_queries", [question]),
        "candidate_top_k": plan.get("candidate_top_k", 40),
        "final_top_k": plan.get("final_top_k", 6),
        "source_strategy": plan.get("source_strategy", "cluster_by_best_source"),
        "answer_style": plan.get("answer_style", "balanced"),
        "planner": plan,
    }
