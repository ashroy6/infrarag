from __future__ import annotations

import json
import os
import re
from typing import Any

from app.llm_client import generate_text

FOLLOWUP_RESOLVER_ENABLED = os.getenv("FOLLOWUP_RESOLVER_ENABLED", "true").lower() == "true"
FOLLOWUP_RESOLVER_TIMEOUT_SECONDS = int(os.getenv("FOLLOWUP_RESOLVER_TIMEOUT_SECONDS", "35"))
FOLLOWUP_RESOLVER_NUM_PREDICT = int(os.getenv("FOLLOWUP_RESOLVER_NUM_PREDICT", "260"))

VAGUE_FOLLOWUP_PATTERNS = [
    r"^elaborate(?:\s+the\s+answer)?\.?$",
    r"^explain\s+more\.?$",
    r"^explain\s+it\s+more\.?$",
    r"^give\s+more\s+details\.?$",
    r"^more\s+details\.?$",
    r"^continue\.?$",
    r"^go\s+on\.?$",
    r"^expand\s+on\s+that\.?$",
    r"^expand\s+it\.?$",
    r"^summarize\s+it\.?$",
    r"^summarise\s+it\.?$",
    r"^what\s+about\s+this\??$",
    r"^what\s+about\s+that\??$",
    r"^tell\s+me\s+more\.?$",
    r"^make\s+it\s+detailed\.?$",
    r"^in\s+detail\.?$",
]


FOLLOWUP_RESOLVER_PROMPT = """
You are the follow-up question resolver for InfraRAG.

Your job:
Rewrite a vague follow-up question into a standalone retrieval question using the recent conversation.

Do NOT answer the user.
Do NOT invent a new topic.
Do NOT mention unrelated older topics.
Use the most recent relevant user question and assistant answer.

Return JSON only.

Rules:
- If the current question is already standalone, return it unchanged.
- If the current question is vague, inherit the topic from the latest relevant user/assistant exchange.
- If the user says "elaborate", "explain more", "continue", or "give more details", rewrite it as a detailed request about the previous answer's topic.
- If the recent context has multiple topics, prefer the immediately previous assistant answer.
- Keep the rewritten question concise and retrieval-friendly.
- Do not include source names unless the user explicitly asked for a source.
- Do not include unrelated entities from older chat turns.

Required JSON:
{{
  "is_followup": true,
  "resolved_question": "standalone retrieval question",
  "reason": "short reason"
}}

Recent conversation:
{chat_context}

Current user question:
{question}
""".strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def looks_like_vague_followup(question: str) -> bool:
    q = normalize_spaces(question).lower()

    if not q:
        return False

    if len(q.split()) <= 4:
        for pattern in VAGUE_FOLLOWUP_PATTERNS:
            if re.match(pattern, q, flags=re.IGNORECASE):
                return True

    if q in {
        "elaborate",
        "continue",
        "explain more",
        "more details",
        "tell me more",
        "in detail",
    }:
        return True

    return False


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


def _fallback_resolve(question: str, chat_context: str) -> dict[str, Any]:
    cleaned = normalize_spaces(question)

    if not looks_like_vague_followup(cleaned):
        return {
            "is_followup": False,
            "resolved_question": cleaned,
            "reason": "Question is standalone.",
        }

    # Cheap deterministic fallback if LLM resolver fails.
    # It does not hardcode topics; it extracts the latest meaningful line from chat context.
    context_lines = [
        normalize_spaces(line)
        for line in (chat_context or "").splitlines()
        if normalize_spaces(line)
    ]

    latest_assistant = ""
    latest_user = ""

    for line in reversed(context_lines):
        lower = line.lower()

        if not latest_assistant and (lower.startswith("assistant:") or lower.startswith("ai:")):
            latest_assistant = re.sub(r"^(assistant|ai):\s*", "", line, flags=re.IGNORECASE)

        if not latest_user and lower.startswith("user:"):
            latest_user = re.sub(r"^user:\s*", "", line, flags=re.IGNORECASE)

        if latest_assistant and latest_user:
            break

    basis = latest_assistant or latest_user

    if basis:
        basis = basis[:350]
        return {
            "is_followup": True,
            "resolved_question": f"Elaborate the previous answer about: {basis}",
            "reason": "Vague follow-up resolved from recent conversation context.",
        }

    return {
        "is_followup": True,
        "resolved_question": cleaned,
        "reason": "Vague follow-up detected but no usable recent context found.",
    }


def resolve_followup_question(question: str, chat_context: str = "") -> dict[str, Any]:
    cleaned = normalize_spaces(question)

    if not FOLLOWUP_RESOLVER_ENABLED:
        return {
            "is_followup": False,
            "resolved_question": cleaned,
            "reason": "Follow-up resolver disabled.",
        }

    if not looks_like_vague_followup(cleaned):
        return {
            "is_followup": False,
            "resolved_question": cleaned,
            "reason": "Question is standalone.",
        }

    if not normalize_spaces(chat_context):
        return {
            "is_followup": True,
            "resolved_question": cleaned,
            "reason": "Vague follow-up detected but no chat context exists.",
        }

    prompt = FOLLOWUP_RESOLVER_PROMPT.format(
        question=cleaned,
        chat_context=chat_context[-5000:],
    )

    try:
        raw = generate_text(
            prompt,
            temperature=0.0,
            num_predict=FOLLOWUP_RESOLVER_NUM_PREDICT,
            timeout=FOLLOWUP_RESOLVER_TIMEOUT_SECONDS,
        )
        data = _extract_json(raw)
    except Exception:
        data = None

    if not data:
        return _fallback_resolve(cleaned, chat_context)

    resolved = normalize_spaces(str(data.get("resolved_question") or cleaned))

    if not resolved:
        resolved = cleaned

    return {
        "is_followup": bool(data.get("is_followup", True)),
        "resolved_question": resolved,
        "reason": str(data.get("reason") or "Resolved by follow-up resolver.")[:300],
    }
