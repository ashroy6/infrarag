from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from app.prompts import VERIFIER_PROMPT

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

VERIFY_COMPLEX_ANSWERS = os.getenv("VERIFY_COMPLEX_ANSWERS", "true").lower() == "true"
VERIFIER_TIMEOUT_SECONDS = int(os.getenv("VERIFIER_TIMEOUT_SECONDS", "90"))
VERIFIER_NUM_PREDICT = int(os.getenv("VERIFIER_NUM_PREDICT", "800"))
VERIFIER_MAX_CONTEXT_CHARS = int(os.getenv("VERIFIER_MAX_CONTEXT_CHARS", "9000"))
VERIFIER_MAX_ANSWER_CHARS = int(os.getenv("VERIFIER_MAX_ANSWER_CHARS", "7000"))

COMPLEX_PIPELINES = {
    "long_explanation",
    "repo_explanation",
    "incident_runbook",
}


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


def _call_ollama_json(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": VERIFIER_NUM_PREDICT,
            },
        },
        timeout=VERIFIER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "") or "")


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
    if not VERIFY_COMPLEX_ANSWERS:
        return False

    if not answer.strip():
        return False

    if answer.strip() == "No evidence found in the knowledge base.":
        return False

    routing = routing or {}
    citations = citations or []

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
) -> dict[str, Any]:
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
        raw = _call_ollama_json(prompt)
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

    if verdict in {"needs_revision", "insufficient_evidence"} and corrected.strip() == draft_answer.strip():
        if verdict == "insufficient_evidence":
            corrected = (
                "The retrieved evidence is insufficient to safely verify the draft answer. "
                "Please ask a narrower question or select a more specific source."
            )
        else:
            corrected = (
                "The draft answer contained claims that could not be safely verified from the retrieved context. "
                "Please ask a narrower question or select a more specific source."
            )

    return {
        "verification_verdict": verdict,
        "unsupported_claims": unsupported_clean,
        "corrected_answer": corrected,
        "verification_reason": str(data.get("reason") or "Verified against retrieved context.")[:500],
        "verification_raw": raw[:1000],
        "verified": verdict in {"valid", "needs_revision", "insufficient_evidence"},
    }
