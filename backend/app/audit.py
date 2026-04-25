from __future__ import annotations

import json
from typing import Any

from app.metadata_db import MetadataDB


def save_audit_event(
    question: str,
    answer: str,
    routing: dict[str, Any],
    citations: list[dict[str, Any]] | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
    conversation_id: str | None = None,
) -> str:
    db = MetadataDB()

    return db.save_audit_log(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        intent=routing.get("intent"),
        pipeline_used=routing.get("pipeline_used"),
        pipeline_label=routing.get("pipeline_label"),
        confidence=float(routing.get("confidence", 0.0)),
        router_reason=routing.get("reason"),
        sources_json=json.dumps(citations or [], ensure_ascii=False),
        model=model,
        latency_ms=latency_ms,
    )
