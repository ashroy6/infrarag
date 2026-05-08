from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

import requests

from app.llm_client import LLMCancelled
from app.prompts import VERIFIER_PROMPT

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

VERIFY_COMPLEX_ANSWERS = os.getenv("VERIFY_COMPLEX_ANSWERS", "true").lower() == "true"
NORMAL_QA_DENIAL_VERIFIER_ENABLED = os.getenv("NORMAL_QA_DENIAL_VERIFIER_ENABLED", "true").lower() == "true"
VERIFIER_TIMEOUT_SECONDS = int(os.getenv("VERIFIER_TIMEOUT_SECONDS", "90"))
VERIFIER_NUM_PREDICT = int(os.getenv("VERIFIER_NUM_PREDICT", "800"))
VERIFIER_MAX_CONTEXT_CHARS = int(os.getenv("VERIFIER_MAX_CONTEXT_CHARS", "9000"))
VERIFIER_MAX_ANSWER_CHARS = int(os.getenv("VERIFIER_MAX_ANSWER_CHARS", "7000"))

COMPLEX_PIPELINES = {
    "long_explanation",
    "repo_explanation",
    "incident_runbook",
}

DENIAL_PHRASES = (
    "no evidence found",
    "no mention",
    "not mentioned",
    "not found",
    "does not mention",
    "doesn't mention",
    "provided context does not",
    "retrieved context does not",
    "context does not support",
    "there is no mention",
    "there are no details",
    "no relevant information",
)


def answer_denies_evidence(answer: str, citations: list[dict[str, Any]] | None = None) -> bool:
    """
    Generic guardrail.

    Returns True when the model denies evidence even though retrieval produced citations.
    This is intentionally not tied to any specific person, company, document, or topic.
    """
    if not citations:
        return False

    text = " ".join((answer or "").lower().split())
    if not text:
        return False

    return any(phrase in text for phrase in DENIAL_PHRASES)


GENERIC_FAILURE_ANSWERS = (
    "the retrieved evidence is insufficient to safely verify the draft answer",
    "please ask a narrower question or select a more specific source",
    "the draft answer contained claims that could not be safely verified",
)


def _looks_like_generic_failure(answer: str) -> bool:
    text = " ".join((answer or "").lower().split())
    return any(phrase in text for phrase in GENERIC_FAILURE_ANSWERS)


def _has_substantive_draft_answer(answer: str) -> bool:
    """
    Generic check: avoid throwing away a useful answer just because verifier is over-strict.
    This does not hardcode any person, company, project, or domain.
    """
    text = " ".join((answer or "").strip().split())
    if not text:
        return False

    lowered = text.lower()
    if _looks_like_generic_failure(lowered):
        return False

    if answer_denies_evidence(text, citations=[{"dummy": "citation_exists"}]):
        return False

    # Needs at least a small factual answer.
    return len(text) >= 25 and len(text.split()) >= 4


def _limit_text(value: str, max_chars: int) -> str:
    text = value or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED FOR VERIFICATION]"


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

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _call_ollama_json_stream(
    prompt: str,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    if cancelled():
        raise LLMCancelled("Verifier cancelled before Ollama request started")

    collected: list[str] = []

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": VERIFIER_NUM_PREDICT,
            },
        },
        timeout=VERIFIER_TIMEOUT_SECONDS,
        stream=True,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if cancelled():
                raise LLMCancelled("Verifier cancelled by user")

            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            token = data.get("response")
            if token:
                collected.append(token)

            if data.get("done") is True:
                break

    return "".join(collected)


def _normalize_verdict(
    value: Any,
    corrected_answer: str,
    draft_answer: str,
    unsupported_claims: list[str],
) -> str:
    verdict = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "valid": "valid",
        "supported": "valid",
        "pass": "valid",
        "passed": "valid",
        "needs_revision": "needs_revision",
        "needs_revisions": "needs_revision",
        "revision_needed": "needs_revision",
        "revise": "needs_revision",
        "revised": "needs_revision",
        "partially_supported": "needs_revision",
        "partially_valid": "needs_revision",
        "insufficient_evidence": "insufficient_evidence",
        "not_enough_evidence": "insufficient_evidence",
        "unsupported": "insufficient_evidence",
        "not_supported": "insufficient_evidence",
    }

    normalized = aliases.get(verdict)

    if normalized:
        if (
            normalized == "insufficient_evidence"
            and corrected_answer.strip()
            and corrected_answer.strip() != draft_answer.strip()
            and corrected_answer.strip() != "No evidence found in the knowledge base."
        ):
            return "needs_revision"
        return normalized

    if corrected_answer.strip() and corrected_answer.strip() != draft_answer.strip():
        return "needs_revision"

    if unsupported_claims:
        return "needs_revision"

    return "not_verified"


def should_verify_answer(
    pipeline_used: str,
    routing: dict[str, Any] | None = None,
    answer: str = "",
    citations: list[dict[str, Any]] | None = None,
) -> bool:
    routing = routing or {}
    citations = citations or []

    if not answer.strip():
        return False

    # Normal Q&A should stay fast and should not be blocked by the verifier.
    # We will handle bad short answers with retrieval trace + targeted recovery later.
    if pipeline_used == "normal_qa":
        return False

    if not VERIFY_COMPLEX_ANSWERS:
        return False

    if answer.strip() == "No evidence found in the knowledge base.":
        return False

    if pipeline_used in COMPLEX_PIPELINES:
        return True

    if routing.get("source_strategy") == "allow_multiple_sources":
        return True

    if len(citations) >= 8:
        return True

    return False


def _fallback_verification(answer: str, reason: str, raw_output: str = "") -> dict[str, Any]:
    return {
        "verification_verdict": "not_verified",
        "unsupported_claims": [],
        "corrected_answer": answer,
        "verification_reason": reason,
        "verification_raw": raw_output[:1000],
        "verified": False,
    }


def verify_answer(
    question: str,
    pipeline_used: str,
    context_text: str,
    draft_answer: str,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if cancel_check and cancel_check():
        raise LLMCancelled("Verifier cancelled before start")

    if not context_text.strip():
        return _fallback_verification(
            draft_answer,
            "No retrieved context available for verification.",
        )

    prompt = VERIFIER_PROMPT.format(
        question=question,
        pipeline_used=pipeline_used,
        context_text=_limit_text(context_text, VERIFIER_MAX_CONTEXT_CHARS),
        draft_answer=_limit_text(draft_answer, VERIFIER_MAX_ANSWER_CHARS),
    )

    try:
        raw = _call_ollama_json_stream(prompt, cancel_check=cancel_check)
    except LLMCancelled:
        raise
    except Exception as exc:
        return _fallback_verification(draft_answer, f"Verifier failed: {exc}")

    data = _extract_json(raw)
    if not data:
        return _fallback_verification(
            draft_answer,
            "Verifier returned invalid JSON even in Ollama JSON mode.",
            raw,
        )

    corrected = str(data.get("corrected_answer") or "").strip()
    if not corrected:
        corrected = draft_answer

    unsupported = data.get("unsupported_claims", [])
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]

    unsupported_clean = [
        str(item)[:500]
        for item in unsupported
        if str(item).strip()
    ]

    verdict = _normalize_verdict(
        data.get("verdict"),
        corrected_answer=corrected,
        draft_answer=draft_answer,
        unsupported_claims=unsupported_clean,
    )

    draft_len = len(draft_answer.strip())
    corrected_len = len(corrected.strip())

    # Guardrail:
    # The verifier must not compress a detailed answer into a tiny summary unless it found concrete unsupported claims.
    # If corrected answer is much shorter and unsupported_claims is empty, keep the original draft as valid.
    if (
        verdict in {"needs_revision", "insufficient_evidence"}
        and not unsupported_clean
        and draft_len >= 500
        and corrected_len < int(draft_len * 0.65)
    ):
        verdict = "valid"
        corrected = draft_answer

    # Guardrail:
    # Do not throw away a useful partial answer and replace it with a generic failure.
    # If the verifier cannot improve the answer, keep the draft when it is substantive.
    if verdict in {"needs_revision", "insufficient_evidence"} and corrected.strip() == draft_answer.strip():
        if _has_substantive_draft_answer(draft_answer):
            verdict = "needs_revision"
            corrected = draft_answer
        elif verdict == "insufficient_evidence":
            corrected = (
                "The retrieved evidence is insufficient to safely answer this. "
                "Please ask a narrower question or select a more specific source."
            )
        else:
            corrected = (
                "The draft answer contained claims that could not be safely verified from the retrieved context. "
                "Please ask a narrower question or select a more specific source."
            )

    # Guardrail:
    # If verifier generates a generic failure but the draft had useful supported content,
    # keep the draft instead of degrading the answer.
    if _looks_like_generic_failure(corrected) and _has_substantive_draft_answer(draft_answer):
        verdict = "needs_revision"
        corrected = draft_answer

    return {
        "verification_verdict": verdict,
        "unsupported_claims": unsupported_clean,
        "corrected_answer": corrected,
        "verification_reason": str(data.get("reason") or "Verified against retrieved context.")[:500],
        "verification_raw": raw[:1000],
        "verified": verdict in {"valid", "needs_revision", "insufficient_evidence"},
    }
