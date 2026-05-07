from __future__ import annotations

import json
from typing import Any

from app.metadata_db import MetadataDB


def make_title_from_question(question: str) -> str:
    clean = " ".join((question or "").strip().split())
    if not clean:
        return "New Chat"
    return clean[:80]


def ensure_conversation(conversation_id: str | None, first_question: str) -> str:
    db = MetadataDB()

    if conversation_id:
        existing = db.get_conversation(conversation_id)
        if existing:
            return conversation_id

    return db.create_conversation(title=make_title_from_question(first_question))


def add_user_message(conversation_id: str, question: str) -> str:
    db = MetadataDB()
    return db.add_chat_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
    )


def add_assistant_message(
    conversation_id: str,
    answer: str,
    citations: list[dict[str, Any]] | None = None,
    intent: str | None = None,
    pipeline_used: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    db = MetadataDB()
    return db.add_chat_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources_json=json.dumps(citations or [], ensure_ascii=False),
        intent=intent,
        pipeline_used=pipeline_used,
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    )


def get_recent_chat_context(conversation_id: str | None, limit: int = 10) -> str:
    if not conversation_id:
        return ""

    db = MetadataDB()
    messages = db.get_recent_chat_messages(conversation_id=conversation_id, limit=limit)

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines)


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    db = MetadataDB()
    return db.list_conversations(limit=limit)


def get_messages(conversation_id: str) -> list[dict[str, Any]]:
    db = MetadataDB()
    return db.get_chat_messages(conversation_id=conversation_id, limit=500)


def rename_conversation(conversation_id: str, title: str) -> None:
    db = MetadataDB()
    db.update_conversation_title(conversation_id=conversation_id, title=title)


def delete_conversation(conversation_id: str) -> None:
    db = MetadataDB()
    db.delete_conversation(conversation_id=conversation_id)
