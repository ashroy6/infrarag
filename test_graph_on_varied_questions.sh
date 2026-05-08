#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/ask}"

QUESTIONS=(
"Who is Mira?"
"Where does the story say \"The light must never be allowed to fail on the seventh winter tide\"?"
"Compare Mira and Elias in The Lantern Keeper of Greyford."
"Summarize The Lantern Keeper of Greyford in 10 lines."
"What is Terraform state?"
"Compare Terraform local state and remote state in AWS."
"Explain the Terraform backend code."
"My Kubernetes pod is in CrashLoopBackOff. What should I check first?"
"Should rollback require approval?"
"What does this corpus say about quantum computing?"
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
import urllib.parse
import urllib.request
import time

api_url = sys.argv[1].rstrip("/")
question = sys.argv[2]

# If your API uses a graph flag in query params, keep graph=true here.
# If your UI-only toggle controls graph server-side, this param will be harmless.
params = urllib.parse.urlencode({"q": question, "graph": "true"})
url = f"{api_url}?{params}"

started = time.time()
try:
    with urllib.request.urlopen(url, timeout=300) as response:
        raw = response.read().decode("utf-8", errors="replace")
except Exception as exc:
    print(f"REQUEST ERROR: {exc}")
    sys.exit(1)

elapsed = round(time.time() - started, 2)

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("NON-JSON RESPONSE:")
    print(raw)
    sys.exit(1)

print(f"Elapsed seconds: {elapsed}")
print(f"Pipeline: {data.get('pipeline_used')}")
print(f"Retriever: {data.get('retriever_used')}")
print(f"Retrieval mode: {data.get('retrieval_mode')}")
print(f"Query shape: {data.get('query_shape')}")
print(f"Reranker: {data.get('reranker_used')}")
print(f"Router: {data.get('router')}")
print(f"Verifier: {data.get('verification_verdict')}")
print(f"Graph: {data.get('graph_context_enabled')}")
print(f"Chunks: {len(data.get('citations') or [])}")
print(f"Latency ms: {data.get('latency_ms')}")

print()
print("ANSWER:")
print((data.get("answer") or "").strip())

print()
citations = data.get("citations") or []
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
