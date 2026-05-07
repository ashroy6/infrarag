#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="debug_refs"
FULL_REF="$OUT_DIR/infrarag_reference_full.txt"
GREP_REF="$OUT_DIR/infrarag_grep_reference.txt"
TREE_REF="$OUT_DIR/infrarag_tree_reference.txt"

mkdir -p "$OUT_DIR"

{
  echo "===== InfraRAG Reference Snapshot ====="
  echo "Generated at: $(date)"
  echo ""

  for f in \
    backend/app/pipeline.py \
    backend/app/retrieve.py \
    backend/app/context_utils.py \
    backend/app/router.py \
    backend/app/qdrant_client.py \
    backend/app/metadata_db.py \
    backend/app/streaming_orchestrator.py \
    backend/app/source_resolver.py \
    backend/app/query_planner.py \
    backend/app/reranker.py \
    frontend/index.html \
    frontend/admin.html \
    frontend/agents.html \
    frontend/graph.html
  do
    if [ -f "$f" ]; then
      echo ""
      echo "===== $f ====="
      cat "$f"
      echo ""
    else
      echo ""
      echo "===== MISSING: $f ====="
      echo ""
    fi
  done
} > "$FULL_REF"

grep -R \
  --exclude='*.bak*' \
  --exclude='*.orig' \
  --exclude='*.rej' \
  --exclude='*.tmp' \
  --exclude='*.pyc' \
  --exclude-dir='__pycache__' \
  "retrieve_context\|candidate_top_k\|final_top_k\|source_id\|chunk_index\|metadata_json\|progress_label\|pipeline_label\|allowed_source_ids\|rerank_hits\|plan_query\|source_strategy" \
  -n backend/app frontend \
  | head -300 > "$GREP_REF" || true

{
  echo "===== InfraRAG Tree Reference ====="
  echo "Generated at: $(date)"
  echo ""
  find backend/app frontend -maxdepth 3 -type f \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -name "*.bak*" \
    ! -name "*.orig" \
    ! -name "*.rej" \
    ! -name "*.tmp" \
    | sort
} > "$TREE_REF"

echo "Created:"
echo "  $FULL_REF"
echo "  $GREP_REF"
echo "  $TREE_REF"
