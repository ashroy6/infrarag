from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))


class LLMError(RuntimeError):
    pass


def generate_text(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    num_predict: int = 512,
    timeout: int | None = None,
) -> str:
    selected_model = model or CHAT_MODEL

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout or OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise LLMError("Ollama request timed out") from exc
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc

    if response.status_code >= 400:
        raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise LLMError(f"Ollama returned non-JSON content-type: {content_type or 'unknown'}")

    try:
        data: dict[str, Any] = response.json()
    except ValueError as exc:
        raise LLMError("Failed to parse Ollama JSON response") from exc

    return (data.get("response") or "").strip()


def stream_generate_text(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    num_predict: int = 512,
    timeout: int | None = None,
) -> Iterator[str]:
    """
    Streams Ollama tokens/chunks.

    Yields text fragments as they arrive.
    """
    selected_model = model or CHAT_MODEL

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout or OLLAMA_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = data.get("response")
                if token:
                    yield token

                if data.get("done") is True:
                    break

    except requests.Timeout as exc:
        raise LLMError("Ollama streaming request timed out") from exc
    except requests.RequestException as exc:
        raise LLMError(f"Ollama streaming request failed: {exc}") from exc
