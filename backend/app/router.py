from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from app.prompts import ROUTER_PROMPT

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
ROUTER_USE_LLM = os.getenv("ROUTER_USE_LLM", "true").lower() == "true"
ROUTER_TIMEOUT_SECONDS = int(os.getenv("ROUTER_TIMEOUT_SECONDS", "20"))

VALID_INTENTS = {
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


def _clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip())


def _fallback(reason: str = "Fallback to normal Q&A.") -> dict[str, Any]:
    return {
        "intent": "normal_qa",
        "pipeline_used": "normal_qa",
        "pipeline_label": PIPELINE_LABELS["normal_qa"],
        "answer_length": "medium",
        "needs_all_chunks": False,
        "confidence": 0.50,
        "reason": reason,
        "router": "fallback",
    }


def _with_label(result: dict[str, Any], router_name: str) -> dict[str, Any]:
    intent = result.get("intent", "normal_qa")
    if intent not in VALID_INTENTS:
        return _fallback(f"Invalid intent returned: {intent}")

    answer_length = result.get("answer_length", "medium")
    if answer_length not in {"short", "medium", "long"}:
        answer_length = "medium"

    return {
        "intent": intent,
        "pipeline_used": intent,
        "pipeline_label": PIPELINE_LABELS[intent],
        "answer_length": answer_length,
        "needs_all_chunks": bool(result.get("needs_all_chunks", False)),
        "confidence": float(result.get("confidence", 0.5)),
        "reason": str(result.get("reason", "Intent classified."))[:300],
        "router": router_name,
    }


def rule_based_intent(question: str) -> dict[str, Any] | None:
    q = _clean_question(question).lower()

    if not q:
        return None

    summary_words = r"\b(summarize|summarise|summary|summarisation|summarization)\b"
    whole_doc_words = r"\b(book|document|pdf|file|source|whole|entire|full|all pages|chapter)\b"

    if re.search(summary_words, q) and re.search(whole_doc_words, q):
        return _with_label(
            {
                "intent": "document_summary",
                "answer_length": "long",
                "needs_all_chunks": True,
                "confidence": 0.95,
                "reason": "User asked for a full document/book/source summary.",
            },
            "rules",
        )

    # Important: long explanation must be checked before repo/Terraform routing.
    # Example: "explain Terraform state locking in detail" is a topic explanation,
    # not necessarily a repo explanation.
    if re.search(r"\b(explain in detail|deep dive|long answer|detailed|step by step|walk me through|in depth|full explanation)\b", q):
        return _with_label(
            {
                "intent": "long_explanation",
                "answer_length": "long",
                "needs_all_chunks": False,
                "confidence": 0.90,
                "reason": "User requested a detailed explanation.",
            },
            "rules",
        )

    if re.search(
        r"\b(error|incident|alert|outage|latency|timeout|crashing|crashloop|pod|failed|failure|down|unhealthy|rollback|what should i check|troubleshoot)\b",
        q,
    ):
        return _with_label(
            {
                "intent": "incident_runbook",
                "answer_length": "medium",
                "needs_all_chunks": False,
                "confidence": 0.88,
                "reason": "User appears to be troubleshooting an operational issue.",
            },
            "rules",
        )

    repo_action = r"\b(explain|summarize|summarise|overview|walk me through|analyse|analyze)\b"
    repo_subject = r"\b(repo|repository|project|codebase|terraform project|terraform repo|tfvars|module|github actions|workflow|pipeline|deployment flow)\b"

    if re.search(repo_action, q) and re.search(repo_subject, q):
        return _with_label(
            {
                "intent": "repo_explanation",
                "answer_length": "long",
                "needs_all_chunks": False,
                "confidence": 0.90,
                "reason": "User asked to explain a repo/project/Terraform/workflow structure.",
            },
            "rules",
        )

    return None


def parse_llm_router_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _fallback("Invalid JSON from Llama router.")

    return _with_label(data, "llm")


def llm_classify_intent(question: str) -> dict[str, Any]:
    prompt = ROUTER_PROMPT.format(question=question)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 180,
                },
            },
            timeout=ROUTER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return parse_llm_router_response(data.get("response", ""))
    except Exception as exc:
        return _fallback(f"Llama router failed: {exc}")


def decide_intent(question: str) -> dict[str, Any]:
    rule_result = rule_based_intent(question)
    if rule_result:
        return rule_result

    if ROUTER_USE_LLM:
        return llm_classify_intent(question)

    return _fallback("No rule matched and Llama router disabled.")
