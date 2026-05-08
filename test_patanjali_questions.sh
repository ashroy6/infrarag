#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/ask}"

QUESTIONS=(
"Who was Patanjali?"
"What does yoga mean in this book?"
"Where does the book say \"Yoga is the control of thought-waves in the mind\"?"
"What are manas, buddhi, and ahamkara?"
"Compare Atman and Purusha according to the book."
"What are the five kinds of thought-waves?"
"What is non-attachment?"
"What are the eight limbs of yoga?"
"Explain the difference between yama and niyama."
"What does this book say about Kubernetes?"
)

i=1
for q in "${QUESTIONS[@]}"; do
  echo
  echo "===================================================================================================="
  echo "TEST $i: $q"
  echo "===================================================================================================="

  python3 - "$API_URL" "$q" <<'PY'
import json
import sys
import urllib.request
import urllib.error
import urllib.parse
import time

api_url = sys.argv[1].rstrip("/")
question = sys.argv[2]


def call_get(url: str, q: str) -> tuple[int, str]:
    encoded = urllib.parse.urlencode({"q": q})
    get_url = f"{url}?{encoded}"

    request = urllib.request.Request(
        get_url,
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, str(exc)


started = time.time()
status, raw = call_get(api_url, question)
elapsed = round(time.time() - started, 2)

if status < 200 or status >= 300:
    print(f"HTTP ERROR: {status}")
    print(raw)
    sys.exit(1)

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("NON-JSON RESPONSE:")
    print(raw)
    sys.exit(1)

print(f"Elapsed seconds: {elapsed}")
print(f"Pipeline: {data.get('pipeline_used')}")
print(f"Mode: {data.get('retrieval_speed') or data.get('mode')}")
print(f"Retriever: {data.get('retriever_used')}")
print(f"Retrieval mode: {data.get('retrieval_mode')}")
print(f"Query shape: {data.get('query_shape')}")
print(f"Reranker: {data.get('reranker_used')}")
print(f"Chunks: {len(data.get('citations') or [])}")
print(f"Planner confidence: {data.get('intent_confidence')}")
print(f"Router: {data.get('router')}")
print(f"Verifier: {data.get('verification_verdict')}")
print(f"Graph: {data.get('graph_context_enabled')}")
print(f"Latency ms: {data.get('latency_ms')}")

print()
print("ANSWER:")
print((data.get("answer") or "").strip())

citations = data.get("citations") or []
print()
print(f"SOURCES / CITATIONS ({len(citations)}):")
for idx, c in enumerate(citations, start=1):
    source = c.get("source") or "unknown"
    chunk = c.get("chunk_index")
    score = c.get("score")
    page = c.get("page_number") or c.get("page_start")
    page_text = f" | Page: {page}" if page is not None else ""
    print(f"{idx}. {source} | Chunk: {chunk}{page_text} | Score: {score}")
PY

  i=$((i + 1))
done
