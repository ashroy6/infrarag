#!/usr/bin/env bash
set -u

BASE_URL="http://localhost:8000/ask"

questions=(
  "From bible for reference, where does it say \"In the beginning\"?"
  "From bible for reference, who created the heaven and the earth?"
  "From bible for reference, explain Psalm 23 in simple terms."
  "From bible for reference, what happened to Jonah?"
  "What is the Yoga Aphorisms of Patanjali document about?"
  "What does the knowledge base say about yoga practice and liberation?"
  "What should be checked during contract review?"
  "What does the NDA sample say about confidentiality obligations?"
  "Which contracts or suppliers are listed in the contract register?"
  "Compare the legal checklist and the supplier services agreement. What risks do they focus on?"
)

for q in "${questions[@]}"; do
  echo "============================================================"
  echo "Query: $q"
  echo "------------------------------------------------------------"

  tmp_body="$(mktemp)"
  http_code="$(
    curl -sG "$BASE_URL" \
      --data-urlencode "q=$q" \
      -o "$tmp_body" \
      -w "%{http_code}"
  )"

  python3 - "$tmp_body" "$http_code" <<'PY'
import json
import sys
from pathlib import Path

body_path = Path(sys.argv[1])
http_code = sys.argv[2]
raw = body_path.read_text(errors="replace").strip()

print("HTTP:", http_code)

if not raw:
    print("EMPTY RESPONSE BODY")
    sys.exit(0)

try:
    data = json.loads(raw)
except Exception as e:
    print("FAILED TO PARSE JSON:", e)
    print()
    print("RAW RESPONSE:")
    print(raw[:3000])
    sys.exit(0)

if "error" in data:
    print("ERROR:")
    print(json.dumps(data["error"], indent=2))
    sys.exit(0)

print("Pipeline:", data.get("pipeline_used"))
print("Retriever:", data.get("retriever_used"))
print("Retrieval mode:", data.get("retrieval_mode"))
print("Query shape:", data.get("query_shape"))
print("Reranker:", data.get("reranker_used"))
print("Latency ms:", data.get("latency_ms"))
print()

print("ANSWER:")
print(data.get("answer", ""))
print()

print("CITATIONS:")
citations = data.get("citations") or []
if not citations:
    print("No citations returned.")
else:
    for i, c in enumerate(citations, start=1):
        print(f"{i}. source={c.get('source')}")
        print(
            "   chunk_index={}, score={}, file_type={}, source_type={}".format(
                c.get("chunk_index"),
                c.get("score"),
                c.get("file_type"),
                c.get("source_type"),
            )
        )
        if c.get("page_number") is not None:
            print("   page={}".format(c.get("page_number")))
        if c.get("structured_reference"):
            print("   structured_reference={}".format(c.get("structured_reference")))
PY

  rm -f "$tmp_body"
  echo
done
