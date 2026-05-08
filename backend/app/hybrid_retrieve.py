from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from app.embedding_service import get_embedding
from app.keyword_index import search_keyword_index
from app.metadata_db import MetadataDB
from app.qdrant_client import get_chunks_by_refs, get_chunks_by_source_id, search
from app.reranker import rerank_hits

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "about", "into", "over", "under", "than", "how", "why", "when", "where",
    "can", "could", "should", "would", "will", "may", "might", "tell", "me",
    "explain", "give", "show", "summarise", "summarize", "please"
}

MAX_KEYWORD_SCAN_SOURCES = int(os.getenv("MAX_KEYWORD_SCAN_SOURCES", "80"))
MAX_KEYWORD_SCAN_CHUNKS_PER_SOURCE = int(os.getenv("MAX_KEYWORD_SCAN_CHUNKS_PER_SOURCE", "3000"))
EXACT_PHRASE_SCAN_LIMIT_PER_SOURCE = int(os.getenv("EXACT_PHRASE_SCAN_LIMIT_PER_SOURCE", "5000"))

ENTITY_SCAN_LIMIT_PER_SOURCE = int(os.getenv("ENTITY_SCAN_LIMIT_PER_SOURCE", "2500"))
ENTITY_MAX_SCAN_SOURCES = int(os.getenv("ENTITY_MAX_SCAN_SOURCES", "80"))
COMPARISON_ENTITY_SCAN_LIMIT_PER_SOURCE = int(os.getenv("COMPARISON_ENTITY_SCAN_LIMIT_PER_SOURCE", "2500"))
COMPARISON_ENTITY_MAX_SCAN_SOURCES = int(os.getenv("COMPARISON_ENTITY_MAX_SCAN_SOURCES", "80"))

ENTITY_IDENTITY_PATTERNS = (
    r"\b{entity}\s+is\s+(?:a|an|the)?\b",
    r"\b{entity}\s+was\s+(?:a|an|the)?\b",
    r"\b{entity}\s*,\s+(?:a|an|the)\b",
    r"\b{entity}\s*[-—]\s*(?:a|an|the)?\b",
    r"\bcalled\s+{entity}\b",
    r"\bnamed\s+{entity}\b",
    r"\bknown\s+as\s+{entity}\b",
    r"\b{entity}'s\s+(?:role|job|identity|purpose|mission|background)\b",
)

ENTITY_NOISE_PATTERNS = (
    r"\bcreated\s+by\s+{entity}\b",
    r"\bopened\s+by\s+{entity}\b",
    r"\bassigned\s+to\s+{entity}\b",
    r"\breviewed\s+by\s+{entity}\b",
    r"\bcommented\s+by\s+{entity}\b",
    r"\bauthor(?:ed)?\s+by\s+{entity}\b",
    r"\bcommitted\s+by\s+{entity}\b",
    r"\bemail\s+from\s+{entity}\b",
    r"\bfrom\s+{entity}\b",
    r"\bissue\s+creator\b",
    r"\bgithub\s+issue\b",
    r"\bpull\s+request\b",
    r"\bcommit\b",
)

BIOGRAPHY_IDENTITY_PATTERNS = (
    r"\bwho\s+(?:is|was)\b",
    r"\blittle\s+is\s+known\b",
    r"\bknown\s+about\b",
    r"\bwas\s+(?:a|an|the)\b",
    r"\bis\s+(?:a|an|the)\b",
    r"\bauthor\b",
    r"\bwrote\b",
    r"\bwritten\s+by\b",
    r"\bcompiler\b",
    r"\bcommentator\b",
    r"\btranslator\b",
    r"\bteacher\b",
    r"\bfounder\b",
    r"\bphilosopher\b",
    r"\bgrammarian\b",
    r"\bscholar\b",
    r"\bdate\b",
    r"\blived\b",
    r"\bborn\b",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def exact_phrase_in_text(text: str, phrase: str) -> bool:
    clean_phrase = re.sub(r"\s+", " ", str(phrase or "").strip())
    if not clean_phrase:
        return False

    words = clean_phrase.split(" ")
    pattern = r"(?<![A-Za-z0-9_])" + r"\s+".join(re.escape(word) for word in words) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, str(text or ""), flags=re.IGNORECASE) is not None


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-/.]+", normalized)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def normalize_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, 1.0))


def source_key(hit: dict[str, Any]) -> str:
    return str(hit.get("source_id") or hit.get("source") or "unknown")


def hit_key(hit: dict[str, Any]) -> tuple[str, int]:
    try:
        chunk_index = int(hit.get("chunk_index", -1))
    except (TypeError, ValueError):
        chunk_index = -1
    return source_key(hit), chunk_index


def dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []

    for hit in hits:
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)

    return deduped


def keyword_score(query: str, text: str, source: str = "", exact_phrases: list[str] | None = None) -> float:
    haystack_text = normalize_text(f"{source}\n{text}")
    query_terms = set(tokenize(query))
    haystack_terms = set(tokenize(haystack_text))

    if not query_terms:
        return 0.0

    matched = query_terms.intersection(haystack_terms)
    overlap = len(matched) / max(len(query_terms), 1)

    exact_bonus = 0.0
    for phrase in exact_phrases or []:
        if exact_phrase_in_text(haystack_text, phrase):
            exact_bonus = max(exact_bonus, 1.0)

    return round(min(1.0, (0.70 * overlap) + (0.30 * exact_bonus)), 6)


def _safe_entity_regex(entity: str) -> str:
    return re.escape(str(entity or "").strip())


def _entity_word_match(text: str, entity: str) -> bool:
    clean_entity = str(entity or "").strip()
    if not clean_entity:
        return False

    pattern = r"(?<![A-Za-z0-9_])" + re.escape(clean_entity) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, str(text or ""), flags=re.IGNORECASE) is not None


def _source_kind_hint(hit: dict[str, Any]) -> str:
    source = str(hit.get("source") or "").lower()
    source_type = str(hit.get("source_type") or "").lower()
    file_type = str(hit.get("file_type") or "").lower()

    combined = f"{source} {source_type} {file_type}"

    if any(marker in combined for marker in ("github", ".git", "issue", "pull_request", "pull-request", "commit")):
        return "github"
    if any(marker in combined for marker in ("ticket", "support", "helpdesk", "incident")):
        return "ticket"
    if any(marker in combined for marker in ("email", "gmail", "mail")):
        return "email"
    if any(marker in combined for marker in ("invoice", "finance")):
        return "finance"
    if file_type in {".md", ".txt", ".pdf", ".docx"}:
        return "document"
    return "unknown"


def _entity_identity_score(text: str, entity: str) -> tuple[float, list[str]]:
    raw = str(text or "")
    if not raw or not entity:
        return 0.0, []

    safe_entity = _safe_entity_regex(entity)
    score = 0.0
    reasons: list[str] = []

    if _entity_word_match(raw, entity):
        score += 0.30
        reasons.append("exact_entity_match")

    entity_count = len(
        re.findall(
            r"(?<![A-Za-z0-9_])" + safe_entity + r"(?![A-Za-z0-9_])",
            raw,
            flags=re.IGNORECASE,
        )
    )
    if entity_count >= 2:
        score += min(0.15, entity_count * 0.03)
        reasons.append("repeated_entity_mentions")

    for pattern_template in ENTITY_IDENTITY_PATTERNS:
        pattern = pattern_template.format(entity=safe_entity)
        if re.search(pattern, raw, flags=re.IGNORECASE):
            score += 0.35
            reasons.append("identity_pattern")
            break

    first_500 = raw[:500]
    if _entity_word_match(first_500, entity):
        score += 0.10
        reasons.append("entity_near_chunk_start")

    for pattern_template in ENTITY_NOISE_PATTERNS:
        pattern = pattern_template.format(entity=safe_entity)
        if re.search(pattern, raw, flags=re.IGNORECASE):
            score -= 0.25
            reasons.append("operational_noise_penalty")
            break

    return round(max(0.0, min(score, 1.0)), 6), reasons


def _entity_source_prior(hit: dict[str, Any]) -> float:
    kind = _source_kind_hint(hit)

    if kind == "document":
        return 0.10
    if kind in {"github", "ticket", "email"}:
        return -0.08
    if kind == "finance":
        return -0.04
    return 0.0


def _looks_like_biographical_question(query: str) -> bool:
    clean = normalize_text(query)
    return bool(re.search(r"^who\s+(?:is|was)\s+", clean))


def _biography_identity_boost(text: str, query: str) -> tuple[float, list[str]]:
    if not _looks_like_biographical_question(query):
        return 0.0, []

    raw = str(text or "")
    reasons: list[str] = []
    boost = 0.0

    for pattern in BIOGRAPHY_IDENTITY_PATTERNS:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            boost += 0.08
            reasons.append("biography_identity_pattern")
            break

    # Foreword/introduction chunks often contain author identity context.
    if re.search(r"\b(foreword|introduction|preface|translator|author|sutras?)\b", raw, flags=re.IGNORECASE):
        boost += 0.08
        reasons.append("intro_author_context")

    return min(boost, 0.18), reasons


def _score_entity_hit_with_query(hit: dict[str, Any], entity: str, query: str) -> dict[str, Any] | None:
    item = _score_entity_hit(hit, entity)
    if not item:
        return None

    bio_boost, bio_reasons = _biography_identity_boost(str(item.get("text") or ""), query)
    if bio_boost:
        item["score"] = round(min(1.0, normalize_score(item.get("score", 0.0)) + bio_boost), 6)
        item["pre_cluster_score"] = item["score"]
        reasons = list(item.get("entity_score_reasons") or [])
        reasons.extend(bio_reasons)
        item["entity_score_reasons"] = reasons
        item["biography_identity_boost"] = bio_boost

    return item


def _score_entity_hit(hit: dict[str, Any], entity: str) -> dict[str, Any] | None:
    text = str(hit.get("text") or "")
    source = str(hit.get("source") or "")

    if not _entity_word_match(f"{source}\n{text}", entity):
        return None

    item = dict(hit)

    lexical = max(
        keyword_score(entity, text, source),
        normalize_score(item.get("keyword_score", 0.0)),
        normalize_score(item.get("lexical_score", 0.0)),
    )
    vector = normalize_score(item.get("vector_score", item.get("score", 0.0)))
    identity, reasons = _entity_identity_score(text, entity)
    source_prior = _entity_source_prior(item)

    final = (0.50 * lexical) + (0.30 * identity) + (0.12 * vector) + source_prior

    if identity >= 0.55:
        final += 0.08

    item["primary_entity"] = entity
    item["entity_identity_score"] = identity
    item["entity_score_reasons"] = reasons
    item["entity_source_kind"] = _source_kind_hint(item)
    item["entity_source_prior"] = source_prior
    item["keyword_score"] = lexical
    item["lexical_score"] = lexical
    item["vector_score"] = vector
    item["pre_cluster_score"] = round(max(0.0, min(final, 1.0)), 6)
    item["score"] = item["pre_cluster_score"]
    item["retrieval_channel"] = item.get("retrieval_channel") or "entity_lookup"

    return item


def _extract_matching_snippet(text: str, phrase: str, radius: int = 260) -> str:
    raw = str(text or "")
    clean_raw = re.sub(r"\s+", " ", raw)
    clean_phrase = re.sub(r"\s+", " ", phrase or "").strip()

    if not clean_raw or not clean_phrase:
        return clean_raw[: radius * 2].strip()

    idx = clean_raw.lower().find(clean_phrase.lower())
    if idx < 0:
        return clean_raw[: radius * 2].strip()

    start = max(0, idx - radius)
    end = min(len(clean_raw), idx + len(clean_phrase) + radius)

    left = clean_raw.rfind(". ", 0, idx)
    if left >= 0 and idx - left < radius:
        start = left + 2

    right_candidates = [
        pos for pos in [
            clean_raw.find(". ", idx + len(clean_phrase)),
            clean_raw.find("\n", idx + len(clean_phrase)),
        ]
        if pos >= 0
    ]
    if right_candidates:
        right = min(right_candidates)
        if right - idx < radius:
            end = right + 1

    snippet = clean_raw[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(clean_raw):
        snippet = snippet + " ..."
    return snippet


def _verify_exact_phrase_hit(hit: dict[str, Any], exact_phrases: list[str]) -> dict[str, Any] | None:
    text = str(hit.get("text") or "")

    for phrase in exact_phrases:
        if exact_phrase_in_text(text, phrase):
            item = dict(hit)
            item["exact_phrase_verified"] = True
            item["matching_phrase"] = phrase
            item["matching_snippet"] = _extract_matching_snippet(text, phrase)
            item["score"] = max(normalize_score(item.get("score", 0.0)), 0.95)
            item["retrieval_channel"] = item.get("retrieval_channel") or "exact_phrase"
            return item

    return None


def _candidate_source_ids(
    *,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[str]:
    if source_id:
        return [source_id]

    db = MetadataDB()
    records = db.list_active_files(source_type=source_type)

    allowed = None
    if allowed_source_ids is not None:
        allowed = {str(item).strip() for item in allowed_source_ids if str(item).strip()}
        if not allowed:
            return []

    source_ids: list[str] = []
    for record in records:
        sid = str(record.get("source_id") or "").strip()
        if not sid:
            continue
        if allowed is not None and sid not in allowed:
            continue
        if file_type and str(record.get("file_type") or "") != file_type:
            continue

        source_ids.append(sid)
        if len(source_ids) >= MAX_KEYWORD_SCAN_SOURCES:
            break

    return source_ids


def _scan_sources_for_exact_phrase(
    *,
    exact_phrases: list[str],
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    source_ids = _candidate_source_ids(
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )

    verified: list[dict[str, Any]] = []

    for sid in source_ids:
        chunks = get_chunks_by_source_id(sid)
        for chunk in chunks[:EXACT_PHRASE_SCAN_LIMIT_PER_SOURCE]:
            item = _verify_exact_phrase_hit(chunk, exact_phrases)
            if item:
                item["retrieval_channel"] = "exact_phrase_scan"
                verified.append(item)

    verified = dedupe_hits(verified)
    verified.sort(
        key=lambda item: (
            str(item.get("source") or ""),
            int(item.get("chunk_index") or 0),
            -normalize_score(item.get("score", 0.0)),
        )
    )
    return verified


def _scan_sources_for_entity(
    *,
    entity: str,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not entity:
        return []

    source_ids = _candidate_source_ids(
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )[:ENTITY_MAX_SCAN_SOURCES]

    hits: list[dict[str, Any]] = []

    for sid in source_ids:
        chunks = get_chunks_by_source_id(sid)
        for chunk in chunks[:ENTITY_SCAN_LIMIT_PER_SOURCE]:
            scored = _score_entity_hit(chunk, entity)
            if not scored:
                continue
            scored["retrieval_channel"] = "entity_exact_scan"
            hits.append(scored)

    hits = dedupe_hits(hits)
    hits.sort(
        key=lambda item: (
            normalize_score(item.get("score", 0.0)),
            normalize_score(item.get("entity_identity_score", 0.0)),
            normalize_score(item.get("keyword_score", 0.0)),
        ),
        reverse=True,
    )
    return hits


SECTION_REF_RE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9_-]{2,50}|chapter|section|part|book|clause|article)\s+(?P<number>\d{1,4})\b",
    re.IGNORECASE,
)

SECTION_REF_STOP_NAMES = {
    "what", "where", "when", "which", "who", "why", "how",
    "find", "show", "give", "tell", "explain", "compare",
}


def _extract_section_references(query: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []

    for match in SECTION_REF_RE.finditer(query or ""):
        name = str(match.group("name") or "").strip()
        number = str(match.group("number") or "").strip()

        if not name or not number:
            continue

        if name.lower() in SECTION_REF_STOP_NAMES:
            continue

        refs.append(
            {
                "name": name,
                "number": number,
                "label": f"{name} {number}",
            }
        )

    return refs[:4]


def _section_ref_in_text(text: str, ref: dict[str, str]) -> bool:
    raw = str(text or "")
    name = re.escape(ref["name"])
    number = re.escape(ref["number"])

    patterns = [
        rf"(?<![A-Za-z0-9_]){name}\s+{number}\s*:",
        rf"(?<![A-Za-z0-9_]){name}\s+{number}\b",
        rf"(?<![A-Za-z0-9_])chapter\s+{number}\b",
        rf"(?<![A-Za-z0-9_])section\s+{number}\b",
        rf"(?<![A-Za-z0-9_]){number}\s*:\s*1\b",
    ]

    name_present = re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", raw, flags=re.IGNORECASE) is not None

    for idx, pattern in enumerate(patterns):
        if idx == len(patterns) - 1 and not name_present:
            continue
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return True

    return False


def _section_matching_snippet(text: str, ref: dict[str, str], radius: int = 1100) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""

    candidates = [
        f"{ref['name']} {ref['number']}:",
        f"{ref['name']} {ref['number']}",
        f"{ref['number']}:1",
    ]

    lower = raw.lower()
    positions = [lower.find(item.lower()) for item in candidates if lower.find(item.lower()) >= 0]
    if not positions:
        return raw[: min(len(raw), radius)].strip()

    idx = min(positions)
    start = max(0, idx - 120)
    end = min(len(raw), idx + radius)

    snippet = raw[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(raw):
        snippet += " ..."
    return snippet


def _scan_sources_for_section_reference(
    *,
    query: str,
    limit: int,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    refs = _extract_section_references(query)
    if not refs:
        return []

    source_ids = _candidate_source_ids(
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )

    hits: list[dict[str, Any]] = []

    for sid in source_ids:
        chunks = get_chunks_by_source_id(sid)

        for chunk in chunks[:EXACT_PHRASE_SCAN_LIMIT_PER_SOURCE]:
            for ref in refs:
                if not _section_ref_in_text(chunk.get("text", ""), ref):
                    continue

                item = dict(chunk)
                item["retrieval_channel"] = "section_reference_scan"
                item["retrieval_mode"] = "section_retrieval"
                item["query_shape"] = "section_summary"
                item["section_reference"] = ref["label"]
                item["matching_snippet"] = _section_matching_snippet(item.get("text", ""), ref)
                item["score"] = max(normalize_score(item.get("score", 0.0)), 0.96)

                text_lower = str(item.get("text", "")).lower()
                label_lower = ref["label"].lower()
                if label_lower in text_lower or f"{label_lower}:" in text_lower:
                    item["score"] = 0.99

                hits.append(item)
                break

    hits = dedupe_hits(hits)
    hits.sort(
        key=lambda item: (
            -normalize_score(item.get("score", 0.0)),
            str(item.get("source") or ""),
            int(item.get("chunk_index") or 0),
        )
    )
    return hits[:limit]


def multi_query_vector_search(
    *,
    queries: list[str],
    candidate_top_k: int,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    all_hits: list[dict[str, Any]] = []

    for query_index, retrieval_query in enumerate(queries):
        query_vector = get_embedding(retrieval_query)

        hits = search(
            query_vector=query_vector,
            limit=candidate_top_k,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
        )

        for hit in hits:
            item = dict(hit)
            vector_score = normalize_score(item.get("score", 0.0))
            lexical_score = keyword_score(retrieval_query, item.get("text", ""), item.get("source", ""))

            item["retrieval_query"] = retrieval_query
            item["retrieval_query_index"] = query_index
            item["retrieval_channel"] = "vector"
            item["vector_score"] = vector_score
            item["lexical_score"] = lexical_score
            item["pre_cluster_score"] = round((0.76 * vector_score) + (0.24 * lexical_score), 6)
            item["score"] = item["pre_cluster_score"]

            all_hits.append(item)

    return dedupe_hits(all_hits)


def keyword_search(
    *,
    query: str,
    queries: list[str],
    keyword_top_k: int,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
    exact_phrases: list[str] | None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    for retrieval_query in queries:
        fts_hits = search_keyword_index(
            query=retrieval_query,
            limit=keyword_top_k,
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
            exact_phrases=exact_phrases,
        )

        for hit in fts_hits:
            text = hit.get("text", "") or ""
            source_path = hit.get("source", "") or ""

            lexical_score = keyword_score(
                retrieval_query,
                text,
                source_path,
                exact_phrases=exact_phrases,
            )
            lexical_score = max(lexical_score, normalize_score(hit.get("score", 0.0)))

            item = dict(hit)
            item["retrieval_query"] = retrieval_query
            item["retrieval_channel"] = "fts_keyword"
            item["keyword_score"] = lexical_score
            item["lexical_score"] = lexical_score
            item["vector_score"] = normalize_score(item.get("vector_score", 0.0))
            item["pre_cluster_score"] = round((0.88 * lexical_score) + (0.12 * normalize_score(item.get("score", 0.0))), 6)
            item["score"] = item["pre_cluster_score"]

            scored.append(item)

    scored = dedupe_hits(scored)
    scored.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)

    return scored[:keyword_top_k]


def source_cluster_scores(hits: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[source_key(hit)].append(hit)

    scores: dict[str, float] = {}

    for sid, items in grouped.items():
        item_scores = sorted(
            [normalize_score(item.get("pre_cluster_score", item.get("score", 0.0))) for item in items],
            reverse=True,
        )
        top_scores = item_scores[:5]
        avg_top = sum(top_scores) / max(len(top_scores), 1)
        max_score = max(item_scores) if item_scores else 0.0
        count_bonus = min(len(items), 6) * 0.025
        keyword_bonus = 0.04 if any(str(item.get("retrieval_channel", "")).startswith("fts") for item in items) else 0.0

        scores[sid] = round((0.62 * avg_top) + (0.26 * max_score) + count_bonus + keyword_bonus, 6)

    return scores


def cluster_sources(hits: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    if not hits:
        return []

    scores = source_cluster_scores(hits)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked:
        return []

    if strategy == "allow_multiple_sources":
        selected_sources = {sid for sid, _ in ranked[:5]}
    elif strategy == "entity_disambiguation":
        selected_sources = {sid for sid, _ in ranked[:4]}
    else:
        selected_sources = {ranked[0][0]}

    clustered: list[dict[str, Any]] = []

    for hit in hits:
        sid = source_key(hit)
        if sid not in selected_sources:
            continue

        item = dict(hit)
        cluster_score = scores.get(sid, 0.0)
        item["source_cluster_score"] = cluster_score
        item["score"] = round((0.78 * normalize_score(item.get("score", 0.0))) + (0.22 * cluster_score), 6)
        clustered.append(item)

    clustered.sort(
        key=lambda item: (
            normalize_score(item.get("score", 0.0)),
            normalize_score(item.get("entity_identity_score", 0.0)),
            normalize_score(item.get("source_cluster_score", 0.0)),
            normalize_score(item.get("keyword_score", 0.0)),
            normalize_score(item.get("vector_score", 0.0)),
        ),
        reverse=True,
    )

    return dedupe_hits(clustered)


def diversify_final_hits(hits: list[dict[str, Any]], top_n: int, strategy: str) -> list[dict[str, Any]]:
    if strategy not in {"allow_multiple_sources", "entity_disambiguation"} or not hits:
        return hits[:top_n]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hit in hits:
        grouped[source_key(hit)].append(hit)

    for source_hits in grouped.values():
        source_hits.sort(
            key=lambda item: (
                normalize_score(item.get("score", 0.0)),
                normalize_score(item.get("entity_identity_score", 0.0)),
            ),
            reverse=True,
        )

    ranked_sources = sorted(
        grouped.keys(),
        key=lambda sid: (
            normalize_score(grouped[sid][0].get("score", 0.0)),
            normalize_score(grouped[sid][0].get("entity_identity_score", 0.0)),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    max_seed_sources = min(4, top_n)
    for sid in ranked_sources[:max_seed_sources]:
        hit = grouped[sid][0]
        key = hit_key(hit)
        if key not in seen:
            seen.add(key)
            selected.append(hit)

    for hit in sorted(
        hits,
        key=lambda item: (
            normalize_score(item.get("score", 0.0)),
            normalize_score(item.get("entity_identity_score", 0.0)),
        ),
        reverse=True,
    ):
        if len(selected) >= top_n:
            break
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)

    selected.sort(
        key=lambda item: (
            normalize_score(item.get("score", 0.0)),
            normalize_score(item.get("entity_identity_score", 0.0)),
        ),
        reverse=True,
    )
    return selected[:top_n]


def expand_neighbour_chunks(hits: list[dict[str, Any]], neighbour_window: int) -> list[dict[str, Any]]:
    if neighbour_window <= 0 or not hits:
        return hits

    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for hit in hits:
        sid = source_key(hit)
        try:
            idx = int(hit.get("chunk_index", -1))
        except (TypeError, ValueError):
            continue

        if not sid or idx < 0:
            continue

        for n in range(idx - neighbour_window, idx + neighbour_window + 1):
            if n < 0:
                continue
            key = (sid, n)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"source_id": sid, "chunk_index": n})

    neighbours = get_chunks_by_refs(refs)

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for hit in hits:
        by_key[hit_key(hit)] = hit

    for chunk in neighbours:
        key = hit_key(chunk)
        if key in by_key:
            continue

        item = dict(chunk)
        item["retrieval_channel"] = "neighbour"
        item["score"] = max(0.01, normalize_score(item.get("score", 0.0)) * 0.85)
        item["neighbour_added"] = True
        by_key[key] = item

    merged = list(by_key.values())
    merged.sort(key=lambda item: normalize_score(item.get("score", 0.0)), reverse=True)
    return merged


def _exact_phrase_retrieve(
    *,
    query: str,
    exact_phrases: list[str],
    limit: int,
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    keyword_hits = keyword_search(
        query=query,
        queries=[query],
        keyword_top_k=max(50, limit * 10),
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
        allowed_source_ids=allowed_source_ids,
        exact_phrases=exact_phrases,
    )

    verified: list[dict[str, Any]] = []
    for hit in keyword_hits:
        item = _verify_exact_phrase_hit(hit, exact_phrases)
        if item:
            verified.append(item)

    scanned = _scan_sources_for_exact_phrase(
        exact_phrases=exact_phrases,
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )
    verified.extend(scanned)

    verified = dedupe_hits(verified)

    verified.sort(
        key=lambda item: (
            str(item.get("source") or ""),
            int(item.get("chunk_index") or 0),
            -normalize_score(item.get("score", 0.0)),
        )
    )

    final_hits = verified[:limit]
    for hit in final_hits:
        hit["adaptive_retrieval"] = True
        hit["retrieval_mode"] = "exact_phrase_search"
        hit["retrieval_speed"] = "direct"
        hit["retriever_used"] = "FTS5 + exact scan"
        hit["query_shape"] = "exact_phrase"
        hit["reranker_used"] = False
        hit["neighbour_window"] = 0
        hit["retrieval_planner_reason"] = "Exact phrase retrieval verified the phrase exists in each returned chunk."

    return final_hits


def _entity_lookup_retrieve(
    *,
    query: str,
    retrieval_plan: dict[str, Any],
    limit: int,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    entity = str(retrieval_plan.get("primary_entity") or "").strip()
    if not entity:
        return []

    safe_limit = max(1, min(int(limit), 50))

    queries = retrieval_plan.get("rewritten_queries") or [entity]
    queries = [
        re.sub(r"\s+", " ", str(item or "").strip())
        for item in queries
        if str(item or "").strip()
    ]

    # For entity lookup, exact/keyword evidence must dominate.
    keyword_hits = keyword_search(
        query=entity,
        queries=queries[:5],
        keyword_top_k=max(40, int(retrieval_plan.get("keyword_top_k", 60) or 60)),
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
        allowed_source_ids=allowed_source_ids,
        exact_phrases=[entity],
    )

    scored_hits: list[dict[str, Any]] = []
    for hit in keyword_hits:
        scored = _score_entity_hit_with_query(hit, entity, query)
        if scored:
            scored["retrieval_channel"] = hit.get("retrieval_channel") or "entity_fts"
            scored_hits.append(scored)

    # Exact scan catches chunks FTS may miss due to tokenisation or chunk shape.
    scanned_hits = _scan_sources_for_entity(
        entity=entity,
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )
    scored_hits.extend(scanned_hits)

    use_vector = bool(retrieval_plan.get("use_vector_search", True))
    entity_is_short_name = len(tokenize(entity)) <= 1 and len(entity) <= 40

    # Vector is useful as fallback, but dangerous for short names. Keep it small.
    if use_vector:
        vector_limit = 20 if entity_is_short_name else 40
        vector_hits = multi_query_vector_search(
            queries=[entity],
            candidate_top_k=vector_limit,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
        )
        for hit in vector_hits:
            scored = _score_entity_hit_with_query(hit, entity, query)
            if scored:
                scored["retrieval_channel"] = "entity_vector"
                # Demote vector-only short-name matches unless they have identity evidence.
                if entity_is_short_name and normalize_score(scored.get("entity_identity_score", 0.0)) < 0.45:
                    scored["score"] = round(normalize_score(scored.get("score", 0.0)) * 0.72, 6)
                    scored["pre_cluster_score"] = scored["score"]
                    scored.setdefault("entity_score_reasons", []).append("short_name_vector_demotion")
                scored_hits.append(scored)

    scored_hits = dedupe_hits(scored_hits)
    if not scored_hits:
        return []

    # Entity lookup must not select one source too early.
    clustered = cluster_sources(scored_hits, "entity_disambiguation")

    # Keep a balanced set before rerank: top identity chunks from several sources.
    pre_rerank = diversify_final_hits(
        clustered,
        top_n=max(int(retrieval_plan.get("final_top_k", safe_limit) or safe_limit) * 2, safe_limit),
        strategy="entity_disambiguation",
    )

    neighbour_window = int(retrieval_plan.get("neighbour_window", 0) or 0)
    if neighbour_window > 0:
        pre_rerank = expand_neighbour_chunks(pre_rerank, neighbour_window=neighbour_window)

        rescored: list[dict[str, Any]] = []
        for hit in pre_rerank:
            scored = _score_entity_hit_with_query(hit, entity, query)
            if scored:
                scored["retrieval_channel"] = hit.get("retrieval_channel") or "entity_neighbour"
                rescored.append(scored)
            elif hit.get("neighbour_added"):
                item = dict(hit)
                item["primary_entity"] = entity
                item["entity_identity_score"] = 0.0
                item["score"] = min(normalize_score(item.get("score", 0.0)), 0.25)
                rescored.append(item)
        pre_rerank = dedupe_hits(rescored)

    use_reranker = bool(retrieval_plan.get("use_reranker", True))
    final_top_k = max(safe_limit, int(retrieval_plan.get("final_top_k", safe_limit) or safe_limit))
    final_top_k = max(1, min(final_top_k, 16))

    if use_reranker:
        reranked = rerank_hits(
            question=query,
            hits=pre_rerank,
            top_n=max(final_top_k * 2, final_top_k),
        )

        # Blend reranker output with deterministic entity evidence.
        for hit in reranked:
            entity_score = normalize_score(hit.get("entity_identity_score", 0.0))
            current_score = normalize_score(hit.get("score", 0.0))
            hit["score"] = round((0.72 * current_score) + (0.28 * entity_score), 6)
    else:
        reranked = sorted(
            pre_rerank,
            key=lambda item: (
                normalize_score(item.get("score", 0.0)),
                normalize_score(item.get("entity_identity_score", 0.0)),
            ),
            reverse=True,
        )

    final_hits = diversify_final_hits(
        reranked,
        top_n=final_top_k,
        strategy="entity_disambiguation",
    )[:safe_limit]

    for hit in final_hits:
        hit["adaptive_retrieval"] = True
        hit["retrieval_mode"] = "entity_lookup"
        hit["retrieval_speed"] = retrieval_plan.get("retrieval_speed", "normal")
        hit["retriever_used"] = "entity FTS/exact scan + guarded Qdrant"
        hit["query_shape"] = retrieval_plan.get("query_shape", "entity_lookup")
        hit["reranker_used"] = bool(use_reranker)
        hit["neighbour_window"] = int(retrieval_plan.get("neighbour_window", 0) or 0)
        hit["retrieval_planner_reason"] = retrieval_plan.get("planner_reason", "")
        hit["primary_entity"] = entity

    return final_hits




def _comparison_score_hit(hit: dict[str, Any], entities: list[str]) -> dict[str, Any] | None:
    if len(entities) < 2:
        return None

    text = str(hit.get("text") or "")
    source = str(hit.get("source") or "")
    haystack = f"{source}\n{text}"

    present = [
        ent for ent in entities
        if _entity_word_match(haystack, ent)
    ]

    if not present:
        return None

    item = dict(hit)
    lexical = max(
        keyword_score(" ".join(entities), text, source),
        normalize_score(item.get("keyword_score", 0.0)),
        normalize_score(item.get("lexical_score", 0.0)),
    )
    vector = normalize_score(item.get("vector_score", item.get("score", 0.0)))

    both_bonus = 0.35 if len(present) >= 2 else 0.0
    single_bonus = 0.12 if len(present) == 1 else 0.0
    proximity_bonus = 0.0

    if len(present) >= 2:
        lowered = haystack.lower()
        positions = []
        for ent in entities[:2]:
            idx = lowered.find(ent.lower())
            if idx >= 0:
                positions.append(idx)
        if len(positions) == 2 and abs(positions[0] - positions[1]) <= 1200:
            proximity_bonus = 0.15

    final = (0.45 * lexical) + (0.20 * vector) + both_bonus + single_bonus + proximity_bonus

    item["comparison_entities"] = entities
    item["comparison_present_entities"] = present
    item["comparison_both_entities"] = len(present) >= 2
    item["keyword_score"] = lexical
    item["lexical_score"] = lexical
    item["vector_score"] = vector
    item["pre_cluster_score"] = round(max(0.0, min(final, 1.0)), 6)
    item["score"] = item["pre_cluster_score"]
    item["retrieval_channel"] = item.get("retrieval_channel") or "comparison"

    return item


def _entity_presence_counts(hits: list[dict[str, Any]], entities: list[str]) -> dict[str, int]:
    counts = {entity: 0 for entity in entities[:2]}
    for hit in hits:
        present = set(hit.get("comparison_present_entities") or [])
        for entity in counts:
            if entity in present:
                counts[entity] += 1
    return counts


def _rescore_comparison_after_rerank(hit: dict[str, Any]) -> dict[str, Any]:
    item = dict(hit)
    base = normalize_score(item.get("score", 0.0))
    keyword = normalize_score(item.get("keyword_score", 0.0))
    vector = normalize_score(item.get("vector_score", 0.0))

    both_bonus = 0.18 if item.get("comparison_both_entities") else 0.0
    single_bonus = 0.06 if item.get("comparison_present_entities") else 0.0

    # Blend reranker/vector score with deterministic lexical/entity evidence.
    item["score"] = round(
        max(0.0, min(1.0, (0.50 * base) + (0.30 * keyword) + (0.10 * vector) + both_bonus + single_bonus)),
        6,
    )
    return item


def _force_balanced_comparison_hits(
    *,
    hits: list[dict[str, Any]],
    entities: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """
    Ensure comparison context contains evidence for both sides.

    Generic behavior:
    - Prefer chunks containing both compared entities.
    - Force evidence for entity A.
    - Force evidence for entity B.
    - Fill the rest by score.
    - Never allow one side to completely dominate when the other side exists.
    """
    if len(entities) < 2 or not hits:
        return hits[:limit]

    rescored = [_rescore_comparison_after_rerank(hit) for hit in hits]
    rescored = dedupe_hits(rescored)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    entity_a, entity_b = entities[0], entities[1]
    minimum_per_entity = 2 if limit >= 6 else 1
    max_both_seed = min(3, max(1, limit // 3))

    both = [h for h in rescored if h.get("comparison_both_entities")]
    both.sort(
        key=lambda h: (
            normalize_score(h.get("keyword_score", 0.0)),
            normalize_score(h.get("score", 0.0)),
            normalize_score(h.get("vector_score", 0.0)),
        ),
        reverse=True,
    )

    # Seed with strongest chunks that mention both entities.
    for hit in both[:max_both_seed]:
        key = hit_key(hit)
        if key not in seen:
            seen.add(key)
            selected.append(hit)

    def add_for_entity(entity: str, needed: int) -> None:
        nonlocal selected, seen

        current = sum(
            1
            for h in selected
            if entity in (h.get("comparison_present_entities") or [])
        )

        if current >= needed:
            return

        entity_hits = [
            h for h in rescored
            if entity in (h.get("comparison_present_entities") or [])
        ]
        entity_hits.sort(
            key=lambda h: (
                bool(h.get("comparison_both_entities")),
                normalize_score(h.get("keyword_score", 0.0)),
                normalize_score(h.get("score", 0.0)),
                normalize_score(h.get("vector_score", 0.0)),
            ),
            reverse=True,
        )

        for hit in entity_hits:
            if current >= needed:
                break
            key = hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            selected.append(hit)
            current += 1

    # Force both sides if evidence exists.
    add_for_entity(entity_a, minimum_per_entity)
    add_for_entity(entity_b, minimum_per_entity)

    # Fill remaining slots by balanced score.
    for hit in sorted(
        rescored,
        key=lambda h: (
            bool(h.get("comparison_both_entities")),
            normalize_score(h.get("keyword_score", 0.0)),
            normalize_score(h.get("score", 0.0)),
            normalize_score(h.get("vector_score", 0.0)),
        ),
        reverse=True,
    ):
        if len(selected) >= limit:
            break
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)

    selected = selected[:limit]

    # Diagnostics for UI/API/debugging.
    counts = _entity_presence_counts(selected, entities)
    weak_entities = [entity for entity, count in counts.items() if count == 0]

    for hit in selected:
        hit["comparison_entity_counts"] = counts
        hit["comparison_weak_entities"] = weak_entities
        hit["comparison_balanced_context"] = not weak_entities

    return selected



def _scan_sources_for_comparison_entities(
    *,
    entities: list[str],
    source_id: str | None,
    source_type: str | None,
    file_type: str | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Generic exact scan for comparison queries.

    Purpose:
    - If vector/FTS retrieval over-focuses on entity A, this scan forces
      candidate evidence for entity B when it exists in the corpus.
    - No domain-specific terms.
    """
    clean_entities = [str(e or "").strip() for e in entities if str(e or "").strip()]
    if len(clean_entities) < 2:
        return []

    source_ids = _candidate_source_ids(
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )[:COMPARISON_ENTITY_MAX_SCAN_SOURCES]

    hits: list[dict[str, Any]] = []

    for sid in source_ids:
        chunks = get_chunks_by_source_id(sid)

        for chunk in chunks[:COMPARISON_ENTITY_SCAN_LIMIT_PER_SOURCE]:
            scored = _comparison_score_hit(chunk, clean_entities)
            if not scored:
                continue

            scored["retrieval_channel"] = "comparison_exact_scan"
            hits.append(scored)

    hits = dedupe_hits(hits)
    hits.sort(
        key=lambda h: (
            bool(h.get("comparison_both_entities")),
            normalize_score(h.get("keyword_score", 0.0)),
            normalize_score(h.get("score", 0.0)),
        ),
        reverse=True,
    )
    return hits


def _comparison_retrieve(
    *,
    query: str,
    retrieval_plan: dict[str, Any],
    limit: int,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
    allowed_source_ids: list[str] | None,
) -> list[dict[str, Any]]:
    entities = [
        str(item or "").strip()
        for item in (retrieval_plan.get("comparison_entities") or [])
        if str(item or "").strip()
    ]

    if len(entities) < 2:
        return []

    safe_limit = max(1, min(int(limit), 50))
    queries = retrieval_plan.get("rewritten_queries") or [query, " ".join(entities), *entities]
    queries = [
        re.sub(r"\s+", " ", str(item or "").strip())
        for item in queries
        if str(item or "").strip()
    ][:5]

    all_hits: list[dict[str, Any]] = []

    keyword_hits = keyword_search(
        query=query,
        queries=queries,
        keyword_top_k=max(60, int(retrieval_plan.get("keyword_top_k", 60) or 60)),
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
        allowed_source_ids=allowed_source_ids,
        exact_phrases=entities,
    )

    for hit in keyword_hits:
        scored = _comparison_score_hit(hit, entities)
        if scored:
            scored["retrieval_channel"] = hit.get("retrieval_channel") or "comparison_fts"
            all_hits.append(scored)

    vector_hits = multi_query_vector_search(
        queries=[" ".join(entities), *entities],
        candidate_top_k=50,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
        allowed_source_ids=allowed_source_ids,
    )

    for hit in vector_hits:
        scored = _comparison_score_hit(hit, entities)
        if scored:
            scored["retrieval_channel"] = "comparison_vector"
            all_hits.append(scored)

    scanned_hits = _scan_sources_for_comparison_entities(
        entities=entities,
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )
    all_hits.extend(scanned_hits)

    all_hits = dedupe_hits(all_hits)
    if not all_hits:
        return []

    # Prefer chunks containing both entities, but keep some single-entity chunks
    # so the answer can compare both sides.
    both = [h for h in all_hits if h.get("comparison_both_entities")]
    singles = [h for h in all_hits if not h.get("comparison_both_entities")]

    both.sort(key=lambda h: normalize_score(h.get("score", 0.0)), reverse=True)
    singles.sort(key=lambda h: normalize_score(h.get("score", 0.0)), reverse=True)

    balanced: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for hit in both[: max(4, safe_limit)]:
        key = hit_key(hit)
        if key not in seen:
            seen.add(key)
            balanced.append(hit)

    for ent in entities[:2]:
        for hit in singles:
            if ent not in hit.get("comparison_present_entities", []):
                continue
            key = hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            balanced.append(hit)
            break

    for hit in both + singles:
        if len(balanced) >= max(safe_limit * 2, 10):
            break
        key = hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        balanced.append(hit)

    neighbour_window = int(retrieval_plan.get("neighbour_window", 0) or 0)
    if neighbour_window > 0:
        balanced = expand_neighbour_chunks(balanced, neighbour_window=neighbour_window)

    use_reranker = bool(retrieval_plan.get("use_reranker", True))
    if use_reranker:
        reranked = rerank_hits(
            question=query,
            hits=balanced,
            top_n=max(safe_limit * 2, 10),
        )
    else:
        reranked = sorted(
            balanced,
            key=lambda h: normalize_score(h.get("score", 0.0)),
            reverse=True,
        )

    final_hits = _force_balanced_comparison_hits(
        hits=reranked,
        entities=entities,
        limit=safe_limit,
    )

    for hit in final_hits:
        hit["adaptive_retrieval"] = True
        hit["retrieval_mode"] = "balanced_multi_entity_hybrid"
        hit["retrieval_speed"] = retrieval_plan.get("retrieval_speed", "normal")
        hit["retriever_used"] = "comparison FTS + Qdrant balanced entities"
        hit["query_shape"] = "comparison"
        hit["reranker_used"] = bool(use_reranker)
        hit["neighbour_window"] = int(retrieval_plan.get("neighbour_window", 0) or 0)
        hit["retrieval_planner_reason"] = retrieval_plan.get("planner_reason", "")
        hit["comparison_entities"] = entities

    return final_hits


def adaptive_hybrid_retrieve(
    *,
    query: str,
    retrieval_plan: dict[str, Any],
    limit: int,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    allowed_source_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))

    queries = retrieval_plan.get("rewritten_queries") or [query]
    queries = [
        re.sub(r"\s+", " ", str(item or "").strip())
        for item in queries
        if str(item or "").strip()
    ]
    if query not in queries:
        queries.insert(0, query)
    queries = queries[:5]

    exact_phrases = [
        str(item or "").strip()
        for item in (retrieval_plan.get("exact_phrases") or [])
        if str(item or "").strip()
    ]

    retrieval_mode = str(retrieval_plan.get("retrieval_mode") or "")

    if retrieval_mode == "exact_phrase_search" and exact_phrases:
        return _exact_phrase_retrieve(
            query=query,
            exact_phrases=exact_phrases,
            limit=safe_limit,
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
        )

    if retrieval_mode == "balanced_multi_entity_hybrid" or retrieval_plan.get("query_shape") == "comparison":
        comparison_hits = _comparison_retrieve(
            query=query,
            retrieval_plan=retrieval_plan,
            limit=safe_limit,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
        )
        if comparison_hits:
            return comparison_hits

    if retrieval_mode == "entity_lookup" or retrieval_plan.get("query_shape") in {"entity_lookup", "definition"}:
        entity_hits = _entity_lookup_retrieve(
            query=query,
            retrieval_plan=retrieval_plan,
            limit=safe_limit,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
            allowed_source_ids=allowed_source_ids,
        )
        if entity_hits:
            return entity_hits

    if retrieval_mode in {"section_retrieval", "section_topic_retrieval"}:
        section_hits = _scan_sources_for_section_reference(
            query=query,
            limit=max(safe_limit, int(retrieval_plan.get("final_top_k", safe_limit) or safe_limit)),
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
            allowed_source_ids=allowed_source_ids,
        )
        if section_hits:
            for hit in section_hits:
                hit["adaptive_retrieval"] = True
                hit["retrieval_mode"] = "section_retrieval"
                hit["retrieval_speed"] = retrieval_plan.get("retrieval_speed", "normal")
                hit["retriever_used"] = "section scan + Qdrant payload"
                hit["query_shape"] = "section_summary"
                hit["reranker_used"] = False
                hit["neighbour_window"] = 0
                hit["retrieval_planner_reason"] = "Section reference scan anchored retrieval to the named chapter/section."
            return section_hits[:safe_limit]

    candidate_top_k = max(safe_limit * 5, int(retrieval_plan.get("candidate_top_k", 40) or 40))
    candidate_top_k = max(10, min(candidate_top_k, 100))

    final_top_k = max(safe_limit, int(retrieval_plan.get("final_top_k", safe_limit) or safe_limit))
    final_top_k = max(1, min(final_top_k, 16))

    keyword_top_k = max(final_top_k, int(retrieval_plan.get("keyword_top_k", 30) or 30))
    keyword_top_k = max(5, min(keyword_top_k, 80))

    source_strategy = str(retrieval_plan.get("source_strategy") or "cluster_by_best_source")
    use_vector = bool(retrieval_plan.get("use_vector_search", True))
    use_keyword = bool(retrieval_plan.get("use_keyword_search", False))
    use_reranker = bool(retrieval_plan.get("use_reranker", True))

    all_hits: list[dict[str, Any]] = []

    if use_vector:
        all_hits.extend(
            multi_query_vector_search(
                queries=queries,
                candidate_top_k=candidate_top_k,
                source_id=source_id,
                source=source,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
                allowed_source_ids=allowed_source_ids,
            )
        )

    if use_keyword:
        all_hits.extend(
            keyword_search(
                query=query,
                queries=queries,
                keyword_top_k=keyword_top_k,
                source_id=source_id,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
                allowed_source_ids=allowed_source_ids,
                exact_phrases=exact_phrases,
            )
        )

    all_hits = dedupe_hits(all_hits)
    if not all_hits:
        return []

    clustered = cluster_sources(all_hits, source_strategy)
    if not clustered:
        return []

    neighbour_window = int(retrieval_plan.get("neighbour_window", 0) or 0)
    if neighbour_window > 0:
        clustered = expand_neighbour_chunks(clustered[:final_top_k], neighbour_window=neighbour_window)

    if use_reranker:
        reranked = rerank_hits(
            question=query,
            hits=clustered,
            top_n=max(final_top_k * 2, final_top_k),
        )
    else:
        reranked = clustered[: max(final_top_k * 2, final_top_k)]

    diversified = diversify_final_hits(
        reranked,
        top_n=final_top_k,
        strategy=source_strategy,
    )

    final_hits = diversified[:safe_limit]

    if use_keyword and use_vector:
        retriever_used = "FTS5 + Qdrant"
    elif use_keyword:
        retriever_used = "FTS5"
    elif use_vector:
        retriever_used = "Qdrant"
    else:
        retriever_used = "none"

    for hit in final_hits:
        hit.setdefault("adaptive_retrieval", True)
        hit.setdefault("retrieval_mode", retrieval_plan.get("retrieval_mode", "vector_rerank"))
        hit.setdefault("retrieval_speed", retrieval_plan.get("retrieval_speed", "normal"))
        hit.setdefault("retriever_used", retriever_used)
        hit.setdefault("query_shape", retrieval_plan.get("query_shape", "normal_qa"))
        hit.setdefault("reranker_used", bool(use_reranker))
        hit.setdefault("neighbour_window", int(retrieval_plan.get("neighbour_window", 0) or 0))
        hit.setdefault("retrieval_planner_reason", retrieval_plan.get("planner_reason", ""))

    return final_hits
