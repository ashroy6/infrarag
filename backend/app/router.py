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


def _clean(q: str) -> str:
    return " ".join((q or "").lower().split())


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




def _source_scoped_question_body(q: str) -> str | None:
    """
    Detect questions like:
    - In bible.txt, who created the heaven and the earth?
    - In bible.txt, summarise Genesis chapter 1.
    - From 01-small-novel.md, who is Mira?

    These are NOT file-explanation requests. They are normal RAG questions
    scoped to one source.
    """
    pattern = (
        r"^\s*(?:in|from|inside|within|using)\s+"
        r"[`'\"]?"
        r"[a-zA-Z0-9_\-./]+"
        r"(?:\.(?:pdf|docx|md|txt|rst|py|js|ts|tsx|jsx|json|yaml|yml|tf|sh|sql|html|css)|dockerfile)"
        r"[`'\"]?"
        r"\s*,?\s+"
        r"(?P<body>.+?)\s*$"
    )

    match = re.match(pattern, q, flags=re.IGNORECASE)
    if not match:
        return None

    body = _clean(match.group("body"))

    content_question_starts = (
        "who ",
        "what ",
        "where ",
        "when ",
        "which ",
        "why ",
        "how ",
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "summarize ",
        "summarise ",
        "summary ",
        "explain ",
        "describe ",
        "compare ",
        "find ",
        "show ",
        "list ",
        "give ",
        "tell ",
        "name ",
    )

    if body.startswith(content_question_starts):
        return body

    return None


def _looks_like_source_scoped_content_question(q: str) -> bool:
    return _source_scoped_question_body(q) is not None


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
        "connection refused",
        "permission denied",
        "access denied",
        "crash",
        "crashed",
        "stuck",
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


def _looks_like_comparison_question(q: str) -> bool:
    comparison_markers = (
        "compare",
        "difference between",
        "differences between",
        " vs ",
        " versus ",
        "better than",
        "pros and cons",
        "advantages and disadvantages",
        "tradeoff",
        "trade-offs",
        "which is better",
    )

    padded = f" {q} "
    return any(marker in padded for marker in comparison_markers)


def _looks_like_yes_no_relationship_question(q: str) -> bool:
    """
    Universal yes/no relationship questions.

    Examples:
    - is CI/CD part of DevOps?
    - are containers used in Kubernetes?
    - does Terraform manage infrastructure?
    - can DevOps include monitoring?
    """
    yes_no_start = (
        r"^is\s+",
        r"^are\s+",
        r"^was\s+",
        r"^were\s+",
        r"^do\s+",
        r"^does\s+",
        r"^did\s+",
        r"^can\s+",
        r"^could\s+",
        r"^should\s+",
        r"^would\s+",
        r"^will\s+",
        r"^has\s+",
        r"^have\s+",
        r"^had\s+",
    )

    if not any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in yes_no_start):
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

    padded = f" {q} "
    return any(marker in padded for marker in relationship_markers) or q.endswith("?")


def _looks_like_how_to_question(q: str) -> bool:
    how_to_patterns = (
        r"^how\s+to\s+",
        r"^how\s+do\s+i\s+",
        r"^how\s+do\s+we\s+",
        r"^how\s+can\s+i\s+",
        r"^how\s+can\s+we\s+",
        r"^how\s+should\s+i\s+",
        r"^how\s+should\s+we\s+",
        r"^steps\s+to\s+",
        r"^best\s+way\s+to\s+",
    )

    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in how_to_patterns)


def _looks_like_definition_question(q: str) -> bool:
    definition_patterns = (
        r"^what\s+is\s+",
        r"^what\s+are\s+",
        r"^define\s+",
        r"^meaning\s+of\s+",
        r"^tell\s+me\s+about\s+",
    )

    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in definition_patterns)


def _looks_like_list_question(q: str) -> bool:
    list_patterns = (
        r"^list\s+",
        r"^show\s+",
        r"^give\s+me\s+",
        r"^name\s+",
        r"^names\s+of\s+",
        r"^what\s+are\s+the\s+names\s+of\s+",
        r"^what\s+are\s+the\s+",
    )

    list_markers = (
        "list of",
        "names of",
        "examples of",
        "types of",
        "tools",
        "services",
        "components",
        "steps",
        "stages",
        "benefits",
    )

    return any(re.search(pattern, q, flags=re.IGNORECASE) for pattern in list_patterns) or any(
        marker in q for marker in list_markers
    )


def _looks_like_source_navigation_question(q: str) -> bool:
    navigation_markers = (
        "where is",
        "where are",
        "where does it say",
        "which file",
        "which document",
        "which source",
        "which page",
        "where mentioned",
        "where is it mentioned",
        "find where",
        "show where",
        "citation for",
        "source for",
    )

    return any(marker in q for marker in navigation_markers)


def _looks_like_section_reference_question(q: str) -> bool:
    clean = _clean(q)
    if not clean:
        return False

    has_ref = bool(
        re.search(r"\b(?:chapter|section|part|book)\s+\d{1,4}\b", clean, flags=re.IGNORECASE)
        or re.search(r"\b[A-Za-z][A-Za-z0-9_-]{2,40}\s+\d{1,4}\b", clean)
    )

    if not has_ref:
        return False

    markers = (
        "what happens",
        "what is in",
        "summarize",
        "summarise",
        "summary",
        "explain",
        "describe",
        "tell me about",
    )

    return any(marker in clean for marker in markers)


def _looks_like_direct_factual_question(q: str) -> bool:
    """
    Fast-path simple factual/document questions.

    These should not call the LLM planner. They only need normal retrieval + answer.
    """
    clean = _clean(q)
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
    if _looks_like_comparison_question(clean):
        return False
    if _looks_like_how_to_question(clean):
        return False

    direct_patterns = (
        r"^who\s+",
        r"^what\s+",
        r"^where\s+",
        r"^when\s+",
        r"^which\s+",
        r"^is\s+",
        r"^are\s+",
        r"^do\s+",
        r"^does\s+",
        r"^can\s+",
        r"^list\s+",
        r"^show\s+",
        r"^give\s+me\s+",
        r"^tell\s+me\s+about\s+",
        r"^tell\s+me\s+",
        r"^name\s+",
        r"^names\s+of\s+",
        r"^what\s+are\s+the\s+names\s+of\s+",
        r"^what\s+are\s+",
        r"^define\s+",
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
    q = _clean(question)

    # 1. Hard rules first. Do not waste time calling Ollama planner for obvious cases.

    # Source-scoped content questions must stay in normal RAG.
    # Example: "In bible.txt, summarise Genesis chapter 1" means:
    # answer from inside bible.txt, not explain the whole bible.txt file.
    if _looks_like_source_scoped_content_question(q):
        if _looks_like_comparison_question(q):
            return _base_response(
                pipeline="long_explanation",
                question=question,
                reason="Rules-first router selected long explanation because the question compares content inside a specific source.",
                confidence=0.9,
                answer_length="long",
                needs_all_chunks=False,
                candidate_top_k=70,
                final_top_k=10,
                source_strategy="allow_multiple_sources",
                question_type="comparison",
            )

        if _looks_like_section_reference_question(q):
            return _base_response(
                pipeline="normal_qa",
                question=question,
                reason="Rules-first router selected normal Q&A because the question asks about a chapter, section, or numbered part inside a specific source.",
                confidence=0.92,
                answer_length="balanced",
                needs_all_chunks=False,
                candidate_top_k=80,
                final_top_k=12,
                source_strategy="cluster_by_best_source",
                question_type="section_summary",
            )

        if _looks_like_topic_summary(q):
            return _base_response(
                pipeline="normal_qa",
                question=question,
                reason="Rules-first router selected normal Q&A because the question asks for a source-scoped topic summary, not a whole-file explanation.",
                confidence=0.9,
                answer_length="balanced",
                needs_all_chunks=False,
                candidate_top_k=60,
                final_top_k=8,
                source_strategy="cluster_by_best_source",
                question_type="topic_summary",
            )

        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question is scoped to a source but asks about content inside it.",
            confidence=0.9,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=8,
            source_strategy="cluster_by_best_source",
            question_type="source_scoped_qa",
        )

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

    if _looks_like_comparison_question(q):
        return _base_response(
            pipeline="long_explanation",
            question=question,
            reason="Rules-first router selected long explanation because the question asks for comparison, trade-offs, or differences.",
            confidence=0.9,
            answer_length="long",
            needs_all_chunks=False,
            candidate_top_k=70,
            final_top_k=10,
            source_strategy="allow_multiple_sources",
            question_type="comparison",
        )

    if _looks_like_section_reference_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks about a chapter, section, or numbered part of a source.",
            confidence=0.9,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=70,
            final_top_k=10,
            source_strategy="cluster_by_best_source",
            question_type="section_summary",
        )

    if _looks_like_how_to_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks for practical how-to guidance.",
            confidence=0.88,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=8,
            source_strategy="cluster_by_best_source",
            question_type="how_to_steps",
        )

    if _looks_like_yes_no_relationship_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks a yes/no relationship or classification.",
            confidence=0.9,
            answer_length="concise",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="yes_no_relationship",
        )

    if _looks_like_source_navigation_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks to locate supporting evidence or source location.",
            confidence=0.9,
            answer_length="concise",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="source_navigation",
        )

    if _looks_like_definition_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks for a definition or entity explanation.",
            confidence=0.9,
            answer_length="concise",
            needs_all_chunks=False,
            candidate_top_k=40,
            final_top_k=6,
            source_strategy="cluster_by_best_source",
            question_type="definition",
        )

    if _looks_like_list_question(q):
        return _base_response(
            pipeline="normal_qa",
            question=question,
            reason="Rules-first router selected normal Q&A because the question asks for a list, names, examples, tools, or components.",
            confidence=0.88,
            answer_length="balanced",
            needs_all_chunks=False,
            candidate_top_k=50,
            final_top_k=8,
            source_strategy="allow_multiple_sources",
            question_type="list_or_examples",
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
