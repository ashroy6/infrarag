from __future__ import annotations

from prometheus_client import Counter, Histogram

RAG_QUESTIONS_TOTAL = Counter(
    "rag_questions_total",
    "Total RAG questions handled by InfraRAG.",
    ["pipeline", "status"],
)

RAG_PIPELINE_TOTAL = Counter(
    "rag_pipeline_total",
    "Total RAG requests by selected pipeline.",
    ["pipeline"],
)

RAG_GRAPH_CONTEXT_ENABLED_TOTAL = Counter(
    "rag_graph_context_enabled_total",
    "Total requests where graph context expansion was enabled.",
    ["pipeline"],
)

RAG_GRAPH_CHUNKS_ADDED_TOTAL = Counter(
    "rag_graph_chunks_added_total",
    "Total graph-neighbour chunks added to retrieved context.",
    ["pipeline"],
)

RAG_ANSWER_LATENCY_SECONDS = Histogram(
    "rag_answer_latency_seconds",
    "End-to-end RAG answer latency in seconds.",
    ["pipeline", "status"],
    buckets=(1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 300, 600),
)

RAG_VERIFIER_TOTAL = Counter(
    "rag_verifier_total",
    "Answer verifier verdict count.",
    ["pipeline", "verdict"],
)

RAG_NO_EVIDENCE_TOTAL = Counter(
    "rag_no_evidence_total",
    "Total RAG requests that returned no evidence.",
    ["pipeline"],
)

RAG_PLANNER_FALLBACK_TOTAL = Counter(
    "rag_planner_fallback_total",
    "Total requests where planner fallback was used.",
    ["reason"],
)

RAG_OLLAMA_TIMEOUT_TOTAL = Counter(
    "rag_ollama_timeout_total",
    "Total Ollama timeout events observed by InfraRAG.",
    ["stage"],
)
