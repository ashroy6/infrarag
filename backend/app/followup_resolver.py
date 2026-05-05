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
    r"^can\s+you\s+elaborate(?:\s+the\s+above)?\??$",
    r"^elaborate\s+(?:on\s+)?(?:the\s+)?above\.?$",
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

POINT_REF_RE = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:please\s+)?"
    r"(?P<action>explain|elaborate|expand|describe|summarize|summarise|tell\s+me\s+about|what\s+is|what\s+does|what\s+about)?"
    r"\s*(?:the\s+)?"
    r"(?P<label>point|section|item|number|bullet|heading|part)\s*"
    r"(?P<num>\d{1,3})"
    r"(?:\s+(?:in\s+detail|more|again))?"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)

BARE_POINT_REF_RE = re.compile(
    r"^\s*(?P<label>point|section|item|number|bullet|heading|part)\s*(?P<num>\d{1,3})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

NUMBERED_ITEM_RE_TEMPLATE = (
    r"(?ms)"
    r"(?:^|\n)\s*"
    r"{num}"
    r"\s*[\.\)]\s+"
    r"(?P<section>.*?)(?=\n\s*\d+\s*[\.\)]\s+|\Z)"
)

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
- If the user refers to a numbered point, section, item, bullet, or heading, rewrite it using that numbered item from the previous assistant answer.
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


def _looks_like_point_reference(question: str) -> bool:
    q = (question or "").strip()
    return bool(POINT_REF_RE.match(q) or BARE_POINT_REF_RE.match(q))


def _extract_point_number(question: str) -> int | None:
    q = (question or "").strip()

    match = POINT_REF_RE.match(q) or BARE_POINT_REF_RE.match(q)
    if not match:
        return None

    try:
        return int(match.group("num"))
    except (TypeError, ValueError):
        return None


def looks_like_vague_followup(question: str) -> bool:
    q = normalize_spaces(question).lower()

    if not q:
        return False

    if _looks_like_point_reference(q):
        return True

    if len(q.split()) <= 8:
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


def _assistant_blocks(chat_context: str) -> list[str]:
    """
    Extract assistant messages from the chat context created by chat_history.py.

    chat_history.py prefixes each stored message with ROLE:, but assistant content
    itself can contain newlines. This parser keeps multiline assistant answers.
    """
    text = chat_context or ""
    pattern = re.compile(
        r"(?ms)(?:^|\n)ASSISTANT:\s*(?P<body>.*?)(?=\nUSER:|\nASSISTANT:|\Z)"
    )
    return [m.group("body").strip() for m in pattern.finditer(text) if m.group("body").strip()]


def _latest_assistant_answer(chat_context: str) -> str:
    blocks = _assistant_blocks(chat_context)
    return blocks[-1] if blocks else ""


def _latest_user_question(chat_context: str) -> str:
    text = chat_context or ""
    pattern = re.compile(
        r"(?ms)(?:^|\n)USER:\s*(?P<body>.*?)(?=\nUSER:|\nASSISTANT:|\Z)"
    )
    matches = [m.group("body").strip() for m in pattern.finditer(text) if m.group("body").strip()]
    return matches[-1] if matches else ""


def _extract_numbered_section(answer: str, number: int) -> str:
    if not answer or number <= 0:
        return ""

    pattern = NUMBERED_ITEM_RE_TEMPLATE.format(num=re.escape(str(number)))
    match = re.search(pattern, answer)

    if not match:
        return ""

    section = match.group("section").strip()

    # Keep enough detail for retrieval, but avoid dumping a huge answer back into the query.
    section = re.sub(r"\n{3,}", "\n\n", section)
    return section[:1200].strip()


def _first_meaningful_line(text: str) -> str:
    for line in (text or "").splitlines():
        clean = normalize_spaces(line)
        if clean:
            return clean
    return normalize_spaces(text)


def _resolve_point_reference(question: str, chat_context: str) -> dict[str, Any] | None:
    point_number = _extract_point_number(question)
    if point_number is None:
        return None

    latest_answer = _latest_assistant_answer(chat_context)
    if not latest_answer:
        return {
            "is_followup": True,
            "resolved_question": normalize_spaces(question),
            "reason": "Numbered follow-up detected but no previous assistant answer was available.",
        }

    section = _extract_numbered_section(latest_answer, point_number)
    if not section:
        return {
            "is_followup": True,
            "resolved_question": normalize_spaces(question),
            "reason": f"Numbered follow-up detected but point {point_number} was not found in the previous assistant answer.",
        }

    topic_hint = _first_meaningful_line(section)

    resolved_question = (
        f"Explain point {point_number} from the previous assistant answer in detail. "
        f"Point {point_number} was: {topic_hint}. "
        f"Use the recent conversation topic and retrieve evidence about this point."
    )

    return {
        "is_followup": True,
        "resolved_question": normalize_spaces(resolved_question),
        "reason": f"Resolved numbered follow-up by mapping point {point_number} to the previous assistant answer.",
    }


def _fallback_resolve(question: str, chat_context: str) -> dict[str, Any]:
    cleaned = normalize_spaces(question)

    point_resolved = _resolve_point_reference(cleaned, chat_context)
    if point_resolved:
        return point_resolved

    if not looks_like_vague_followup(cleaned):
        return {
            "is_followup": False,
            "resolved_question": cleaned,
            "reason": "Question is standalone.",
        }

    latest_assistant = _latest_assistant_answer(chat_context)
    latest_user = _latest_user_question(chat_context)

    basis = latest_assistant or latest_user

    if basis:
        basis = normalize_spaces(basis)[:500]
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

    # Deterministic handling first. Do not call Ollama for numbered references.
    # This avoids planner/follow-up timeout and fixes questions like:
    # "explain point 6", "elaborate section 3", "what about bullet 2".
    point_resolved = _resolve_point_reference(cleaned, chat_context)
    if point_resolved and point_resolved.get("resolved_question") != cleaned:
        return point_resolved

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
