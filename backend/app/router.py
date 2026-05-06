from __future__ import annotations

import re
from typing import Any

from app.query_planner import PIPELINE_LABELS, VALID_PIPELINES, plan_query


def _safe_label(pipeline: str) -> str:
    return PIPELINE_LABELS.get(pipeline, pipeline)


def _base_response(
    *,
    pipeline: str,
    question: str,
    reason: str,
    confidence: float = 0.9,
    router: str = "rules_first",
    answer_length: str = "balanced",
    needs_all_chunks: bool = False,
    candidate_top_k: int = 40,
    final_top_k: int = 6,
    source_strategy: str = "cluster_by_best_source",
    question_type: str | None = None,
    rewritten_queries: list[str] | None = None,
) -> dict[str, Any]:
    if pipeline not in VALID_PIPELINES:
        pipeline = "normal_qa"

    queries = rewritten_queries or [question]

    return {
        "intent": pipeline,
        "pipeline_used": pipeline,
        "pipeline_label": _safe_label(pipeline),
        "answer_length": answer_length,
        "needs_all_chunks": needs_all_chunks,
        "confidence": confidence,
        "reason": reason[:300],
        "router": router,
        "question_type": question_type,
        "rewritten_queries": queries,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "source_strategy": source_strategy,
        "answer_style": answer_length,
        "planner": {
            "pipeline_used": pipeline,
            "pipeline": pipeline,
            "answer_style": answer_length,
            "needs_full_document": needs_all_chunks,
            "confidence": confidence,
            "reason": reason,
            "router": router,
            "question_type": question_type,
            "rewritten_queries": queries,
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
            "source_strategy": source_strategy,
        },
    }


def _looks_like_file_question(q: str) -> bool:
    file_markers = (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".tf",
        ".sh",
        ".sql",
        ".html",
        ".css",
        ".dockerfile",
        "dockerfile",
    )

    action_markers = (
        "explain",
        "line by line",
        "walk through",
        "what does",
        "how does",
        "improve",
        "review",
        "summarize",
        "summarise",
        "analyse",
        "analyze",
        "describe",
    )

    return any(marker in q for marker in file_markers) and any(
        marker in q for marker in action_markers
    )


def _looks_like_repo_question(q: str) -> bool:
    repo_markers = (
        "repo",
        "repository",
        "project",
        "codebase",
        "folder structure",
        "directory structure",
        "pipeline",
        "workflow",
        "architecture",
        "how is this built",
        "how does this app work",
        "explain the app",
        "explain this app",
        "explain the project",
        "explain this project",
    )

    action_markers = (
        "explain",
        "summarize",
        "summarise",
        "describe",
        "walk through",
        "what is",
        "how does",
        "architecture",
    )

    return any(marker in q for marker in repo_markers) and any(
        marker in q for marker in action_markers
    )


def _looks_like_incident_question(q: str) -> bool:
    incident_markers = (
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
        "troubleshoot",
        "debug",
        "why is this happening",
        "request failed",
    )

    return any(marker in q for marker in incident_markers)


def _looks_like_summary_question(q: str) -> bool:
    """
    Detect full-document, full-book, or full-source summary requests.

    These should use document_summary because the user is asking to summarize
    a complete source, not just a short topic.
    """
    summary_markers = (
        "summarize this book",
        "summarise this book",
        "summarize the book",
        "summarise the book",
        "summarize this document",
        "summarise this document",
        "summarize the document",
        "summarise the document",
        "summarize full document",
        "summarise full document",
        "summary of this document",
        "summary of the document",
        "in 100 lines",
        "in 50 lines",
        "in 20 lines",
        "full summary",
        "whole book",
        "entire book",
        "whole document",
        "entire document",
    )

    return any(marker in q for marker in summary_markers)


def _looks_like_topic_summary(q: str) -> bool:
    """
    Detect short topic-summary requests and route them without calling the LLM planner.

    These are not full-document/book summaries. They should use normal_qa unless
    the user explicitly asks to summarize an entire document, book, source, or file.
    """
    return bool(
        re.match(r"^\s*(summarize|summarise|summary\s+of)\s+.+", q, flags=re.IGNORECASE)
    )


def _looks_like_long_explanation(q: str) -> bool:
    long_markers = (
        "explain in detail",
        "elaborate",
        "elaborate the above",
        "elaborate on the above",
        "can you elaborate",
        "explain more",
        "explain it more",
        "give more details",
        "more details",
        "tell me more",
        "long explanation",
        "detailed explanation",
        "deep dive",
        "step by step",
        "in detail",
    )

    return any(marker in q for marker in long_markers)


def _looks_like_numbered_followup(q: str) -> bool:
    return bool(
        re.match(
            r"^\s*(can\s+you\s+)?(please\s+)?"
            r"(explain|elaborate|expand|describe|summarize|summarise|tell\s+me\s+about|what\s+is|what\s+about)?\s*"
            r"(the\s+)?(point|section|item|number|bullet|heading|part)\s*\d{1,3}"
            r"(\s+(in\s+detail|more|again))?\s*[?.!]*\s*$",
            q,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_direct_factual_question(q: str) -> bool:
    """
    Fast-path simple factual/document questions.

    These should not call the LLM planner. They only need normal retrieval + answer.
    Generic examples:
    - who is X
    - what is X
    - where is X
    - list the names of X
    - tell me about X from the CV/document
    - what are the names/items/tools/projects/roles in source
    """
    clean = " ".join((q or "").lower().split())
    if not clean:
        return False

    # Do not steal obvious specialist routes.
    if _looks_like_summary_question(clean):
        return False
    if _looks_like_long_explanation(clean):
        return False
    if _looks_like_incident_question(clean):
        return False
    if _looks_like_file_question(clean):
        return False

    direct_patterns = (
        r"^who\s+",
        r"^what\s+",
        r"^where\s+",
        r"^when\s+",
        r"^which\s+",
        r"^list\s+",
        r"^show\s+",
        r"^give\s+me\s+",
        r"^tell\s+me\s+about\s+",
        r"^tell\s+me\s+",
        r"^name\s+",
        r"^names\s+of\s+",
        r"^what\s+are\s+the\s+names\s+of\s+",
        r"^what\s+are\s+",
    )

    if any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in direct_patterns):
        return True

    factual_markers = (
        "from the cv",
        "in the cv",
        "from this cv",
        "from the resume",
        "in the resume",
        "from this resume",
        "from the document",
        "in the document",
        "from this document",
        "names of",
        "name of",
        "list of",
        "github repo",
        "github repos",
        "github repositories",
        "experience at",
        "experience in",
    )

    return any(marker in clean for marker in factual_markers)




def decide_intent(question: str, chat_context: str = "") -> dict[str, Any]:
    q = " ".join((question or "").lower().split())

    # 1. Hard rules first. Do not waste time calling Ollama planner for obvious cases.

    if _looks_like_file_question(q):
        return _base_response(
            pipeline="repo_explanation",
            question=question,
            reason="Rules-first router selected repo explanation because the question asks about a specific source/code file.",
            confidence=0.95,
            answer_length="detailed" if "line by line" in q else "balanced",
            needs_all_chunks=True,
            candidate_top_k=60,
            final_top_k=12,
            source_strategy="exact_or_best_source",
            question_type="file_explanation",
        )

    if _looks_like_numbered_followup(q):
        return _base_response(
            pipeline="long_explanation",
            question=question,
            reason="Rules-first router selected long explanation because the question refers to a numbered point or section from the previous answer.",
            confidence=0.9,
            answer_length="long",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=10,
            source_strategy="cluster_by_best_source",
            question_type="numbered_followup",
        )

    if _looks_like_long_explanation(q):
        return _base_response(
            pipeline="long_explanation",
            question=question,
            reason="Rules-first router selected long explanation because the question asks for detailed elaboration.",
            confidence=0.9,
            answer_length="long",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=10,
            source_strategy="cluster_by_best_source",
            question_type="long_explanation",
        )

    if _looks_like_repo_question(q):
        return _base_response(
            pipeline="repo_explanation",
            question=question,
            reason="Rules-first router selected repo explanation because the question asks to explain a repo, project, workflow, architecture, or pipeline.",
            confidence=0.9,
            answer_length="detailed",
            needs_all_chunks=False,
            candidate_top_k=60,
            final_top_k=10,
            source_strategy="cluster_by_best_source",
            question_type="repo_explanation",
        )

    if _looks_like_incident_question(q):
        return _base_response(
            pipeline="incident_runbook",
            question=question,
            reason="Rules-first router selected incident/runbook troubleshooting because the question contains an error, failure, timeout, or debugging signal.",
            confidence=0.9,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=8,
            source_strategy="cluster_by_best_source",
            question_type="incident_or_debugging",
        )

    if _looks_like_summary_question(q):
        return _base_response(
            pipeline="document_summary",
            question=question,
            reason="Rules-first router selected document summary because the question asks for a full document or book summary.",
            confidence=0.9,
            answer_length="long",
            needs_all_chunks=True,
            candidate_top_k=80,
            final_top_k=20,
            source_strategy="single_source_all_chunks",
            question_type="document_summary",
        )

    if _looks_like_topic_summary(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks for a short topic summary, not a full document summary.",
            confidence=0.85,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="topic_summary",
        )

    if _looks_like_direct_factual_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question is a direct factual lookup.",
            confidence=0.9,
            answer_length="concise",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="direct_factual",
        )

    # 2. Only use LLM planner when hard rules do not clearly identify the intent.

    try:
        plan = plan_query(question, chat_context=chat_context)
    except Exception as exc:
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason=f"Planner failed, so router used normal Q&A fallback: {type(exc).__name__}",
            confidence=0.5,
            router="planner_fallback",
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="normal_qa",
        )

    pipeline = plan.get("pipeline_used") or plan.get("pipeline") or "normal_qa"
    if pipeline not in VALID_PIPELINES:
        pipeline = "normal_qa"

    answer_style = plan.get("answer_style", "balanced")

    return {
        "intent": pipeline,
        "pipeline_used": pipeline,
        "pipeline_label": _safe_label(pipeline),
        "answer_length": answer_style,
        "needs_all_chunks": bool(plan.get("needs_full_document", False)),
        "confidence": float(plan.get("confidence", 0.5)),
        "reason": str(plan.get("reason", "Planned by local LLM."))[:300],
        "router": plan.get("router", "llm_planner"),
        "question_type": plan.get("question_type"),
        "rewritten_queries": plan.get("rewritten_queries", [question]),
        "candidate_top_k": plan.get("candidate_top_k", 40),
        "final_top_k": plan.get("final_top_k", 6),
        "source_strategy": plan.get("source_strategy", "cluster_by_best_source"),
        "answer_style": answer_style,
        "planner": plan,
    }
