from __future__ import annotations

import os
import re
from typing import Any

from app.answer_verifier import answer_denies_evidence, should_verify_answer, verify_answer
from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import DENIAL_RECOVERY_PROMPT, NORMAL_QA_PROMPT
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context
from app.source_resolver import resolve_source_for_question

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))
NORMAL_QA_LIMIT = int(os.getenv("NORMAL_QA_LIMIT", "6"))
NORMAL_QA_NUM_PREDICT = int(os.getenv("NORMAL_QA_NUM_PREDICT", "500"))

QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def _chunk_meta(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        return {
            "retriever_used": None,
            "retrieval_mode": None,
            "query_shape": None,
            "reranker_used": None,
            "retrieval_speed": None,
            "primary_entity": None,
        }

    first = chunks[0]
    return {
        "retriever_used": first.get("retriever_used"),
        "retrieval_mode": first.get("retrieval_mode"),
        "query_shape": first.get("query_shape"),
        "reranker_used": first.get("reranker_used"),
        "retrieval_speed": first.get("retrieval_speed"),
        "primary_entity": first.get("primary_entity"),
    }


def _important_query_terms(question: str) -> set[str]:
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
        "where", "when", "how", "why", "does", "do", "did", "this", "that",
        "book", "document", "file", "say", "says", "about", "from", "in",
        "according", "to", "of", "and", "or", "me", "tell", "explain",
    }
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", question or "")
    return {t.lower() for t in terms if t.lower() not in stop}


def _retrieval_looks_like_noise(question: str, chunks: list[dict[str, Any]]) -> bool:
    """
    Hard no-evidence guard.

    If a query has an important rare/proper term and none of the retrieved chunks
    contain it, do not send random context to the LLM.
    """
    if not chunks:
        return True

    terms = _important_query_terms(question)
    if not terms:
        return False

    joined = "\n".join(str(c.get("text") or "") for c in chunks).lower()
    source_joined = "\n".join(str(c.get("source") or "") for c in chunks).lower()
    haystack = joined + "\n" + source_joined

    matched = {term for term in terms if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", haystack)}

    # For a one-term query like Kubernetes, exact absence means no evidence.
    if len(terms) == 1 and not matched:
        return True

    # For multi-term queries, require at least one meaningful term in evidence.
    if not matched:
        return True

    scores = [float(c.get("score", 0.0) or 0.0) for c in chunks]
    if len(scores) >= 4:
        spread = max(scores) - min(scores)
        # Flat mid scores usually mean vector fallback noise.
        if spread < 0.01 and max(scores) < 0.60 and len(matched) < max(1, len(terms) // 2):
            return True

    return False


def _with_meta(payload: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {**payload, **_chunk_meta(chunks)}


def _quoted_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    for match in QUOTE_RE.finditer(question or ""):
        value = (match.group(1) or match.group(2) or "").strip()
        if value:
            phrases.append(value)
    return phrases


def _is_exact_phrase_result(question: str, chunks: list[dict[str, Any]]) -> bool:
    if not _quoted_phrases(question):
        return False
    return any(
        chunk.get("retrieval_mode") == "exact_phrase_search"
        or chunk.get("exact_phrase_verified")
        or chunk.get("query_shape") == "exact_phrase"
        for chunk in chunks
    )


def _format_location(chunk: dict[str, Any]) -> str:
    location_parts: list[str] = []

    if chunk.get("section_title"):
        location_parts.append(f"Section: {chunk.get('section_title')}")

    if chunk.get("page_number") is not None:
        location_parts.append(f"Page: {chunk.get('page_number')}")
    elif chunk.get("page_start") is not None and chunk.get("page_end") is not None:
        location_parts.append(f"Pages: {chunk.get('page_start')}-{chunk.get('page_end')}")

    if chunk.get("record_type"):
        location_parts.append(f"Record: {chunk.get('record_type')}")

    location_parts.append(f"Chunk: {chunk.get('chunk_index', -1)}")

    return " | ".join(location_parts)


def _direct_exact_phrase_answer(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    phrases = _quoted_phrases(question)
    phrase = phrases[0] if phrases else "requested phrase"

    verified_chunks = [
        chunk for chunk in chunks
        if chunk.get("exact_phrase_verified") or chunk.get("matching_snippet")
    ]

    if not verified_chunks:
        verified_chunks = chunks

    compacted = compact_chunks(verified_chunks, max_chars_per_chunk=900, max_total_chars=5000)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    lines = [f'Found exact phrase: "{phrase}"', ""]

    for idx, chunk in enumerate(compacted, start=1):
        source = chunk.get("source", "unknown")
        location = _format_location(chunk)
        snippet = (
            chunk.get("matching_snippet")
            or chunk.get("text", "")
        ).strip()

        if len(snippet) > 900:
            snippet = snippet[:900].rstrip() + " ..."

        lines.append(f"{idx}. {source}")
        lines.append(f"   {location}")
        lines.append(f"   {snippet}")
        lines.append("")

    lines.append("Open the citations for the full evidence.")

    return {
        "answer": "\n".join(lines).strip(),
        "citations": citations,
        "verification_context_text": context_text,
        "retrieval_mode": "exact_phrase_search",
        "query_shape": "exact_phrase",
        "verification_verdict": "skipped_direct",
        "unsupported_claims": [],
        "verification_reason": "Exact phrase lookup was answered deterministically from verified retrieved text.",
        "verified": False,
    }


def run(
    question: str,
    chat_context: str = "",
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    source_resolution = resolve_source_for_question(
        question,
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
    )

    resolved_source_id = source_id or source_resolution.get("source_id")

    chunks = retrieve_context(
        question,
        limit=NORMAL_QA_LIMIT,
        source_id=resolved_source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    if not chunks:
        return _with_meta(no_evidence_response(), chunks)

    if _is_exact_phrase_result(question, chunks):
        return _direct_exact_phrase_answer(question, chunks)

    if _retrieval_looks_like_noise(question, chunks):
        return _with_meta(no_evidence_response(), chunks)

    top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return _with_meta(no_evidence_response(), chunks)

    compacted = compact_chunks(chunks, max_total_chars=7000)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    prompt = NORMAL_QA_PROMPT.format(
        question=question,
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=NORMAL_QA_NUM_PREDICT,
    )

    verification_result: dict[str, Any] = {
        "verification_verdict": "skipped",
        "unsupported_claims": [],
        "verification_reason": "Verifier skipped for this answer.",
        "verified": False,
    }

    if should_verify_answer(
        pipeline_used="normal_qa",
        routing={"source_strategy": "cluster_by_best_source"},
        answer=answer,
        citations=citations,
    ):
        verification_result = verify_answer(
            question=question,
            pipeline_used="normal_qa",
            context_text=context_text,
            draft_answer=answer,
        )

        verdict = verification_result.get("verification_verdict")
        corrected = str(verification_result.get("corrected_answer") or "").strip()

        if verdict in {"needs_revision", "insufficient_evidence"} and corrected:
            answer = corrected

    if answer_denies_evidence(answer, citations):
        recovery_prompt = DENIAL_RECOVERY_PROMPT.format(
            question=question,
            context_text=context_text,
        )
        recovered = generate_text(
            recovery_prompt,
            temperature=0.0,
            num_predict=NORMAL_QA_NUM_PREDICT,
        ).strip()

        if recovered and not answer_denies_evidence(recovered, citations):
            answer = recovered
            verification_result = {
                **verification_result,
                "verification_verdict": "recovered",
                "verification_reason": "Second-pass evidence recovery corrected a false no-evidence answer.",
                "verified": False,
            }

    if answer.strip() == "No evidence found in the knowledge base." and citations:
        return {
            "answer": (
                "Relevant evidence was retrieved, but the answer model failed to extract a supported answer from it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            ),
            "citations": citations,
            "verification_context_text": context_text,
            "verification_verdict": "answer_generation_failed_with_evidence",
            "unsupported_claims": [],
            "verification_reason": "Model returned no-evidence despite retrieved citations.",
            "verified": False,
            **_chunk_meta(compacted),
        }

    if answer_denies_evidence(answer, citations) and citations:
        return {
            "answer": (
                "Relevant evidence was retrieved, but the answer model denied or failed to use it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            ),
            "citations": citations,
            "verification_context_text": context_text,
            "verification_verdict": "answer_denied_retrieved_evidence",
            "unsupported_claims": [],
            "verification_reason": "Model denied evidence despite retrieved citations.",
            "verified": False,
            **_chunk_meta(compacted),
        }

    if answer.strip() == "No evidence found in the knowledge base.":
        return _with_meta(no_evidence_response(), chunks)

    if answer_denies_evidence(answer, citations):
        return _with_meta(no_evidence_response(), chunks)

    return {
        "answer": answer,
        "citations": citations,
        "verification_context_text": context_text,
        "source_resolution": source_resolution,
        **_chunk_meta(compacted),
        **verification_result,
    }
