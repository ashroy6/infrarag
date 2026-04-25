from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass


@dataclass
class CancelState:
    request_id: str
    created_at: float
    cancelled: bool = False
    reason: str = ""


_LOCK = threading.Lock()
_REQUESTS: dict[str, CancelState] = {}
_TTL_SECONDS = 3600


def create_request_id() -> str:
    return str(uuid.uuid4())


def register_request(request_id: str | None = None) -> str:
    rid = request_id or create_request_id()

    with _LOCK:
        _REQUESTS[rid] = CancelState(
            request_id=rid,
            created_at=time.time(),
        )

    cleanup_old_requests()
    return rid


def cancel_request(request_id: str, reason: str = "Cancelled by user.") -> bool:
    if not request_id:
        return False

    with _LOCK:
        state = _REQUESTS.get(request_id)
        if not state:
            _REQUESTS[request_id] = CancelState(
                request_id=request_id,
                created_at=time.time(),
                cancelled=True,
                reason=reason,
            )
            return True

        state.cancelled = True
        state.reason = reason
        return True


def is_cancelled(request_id: str | None) -> bool:
    if not request_id:
        return False

    with _LOCK:
        state = _REQUESTS.get(request_id)
        return bool(state and state.cancelled)


def cancel_reason(request_id: str | None) -> str:
    if not request_id:
        return ""

    with _LOCK:
        state = _REQUESTS.get(request_id)
        return state.reason if state else ""


def unregister_request(request_id: str | None) -> None:
    if not request_id:
        return

    with _LOCK:
        _REQUESTS.pop(request_id, None)


def cleanup_old_requests() -> None:
    cutoff = time.time() - _TTL_SECONDS

    with _LOCK:
        old_keys = [
            request_id
            for request_id, state in _REQUESTS.items()
            if state.created_at < cutoff
        ]

        for request_id in old_keys:
            _REQUESTS.pop(request_id, None)
