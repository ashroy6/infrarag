from __future__ import annotations

import re
from hashlib import sha1
from typing import Any

NUMBERED_REF_RE = re.compile(r"(?<!\d)(?P<section>\d{1,4})\s*[:.]\s*(?P<subsection>\d{1,4})(?!\d)")

NAMED_NUMBER_RE = re.compile(
    r"\b(?P<context>[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,}){0,3})\s+"
    r"(?:chapter\s+|section\s+|article\s+|clause\s+|part\s+)?"
    r"(?P<section>\d{1,4})(?:\s*[:.]\s*(?P<subsection>\d{1,4}))?\b",
    re.IGNORECASE,
)

EXPLICIT_REF_RE = re.compile(
    r"\b(?P<type>chapter|section|article|clause|part|appendix|page)\s+"
    r"(?P<section>[A-Za-z0-9][A-Za-z0-9_.-]{0,20})\b",
    re.IGNORECASE,
)

QUERY_LEADING_NOISE = re.compile(
    r"^(?:in|from|inside|within|using|according\s+to)\s+[`'\"]?[^,]+?[`'\"]?\s*,?\s+",
    re.IGNORECASE,
)

QUESTION_WORDS = {
    "who", "what", "where", "when", "which", "why", "how", "find", "show",
    "give", "tell", "explain", "describe", "summarise", "summarize", "compare",
    "list", "name", "does", "do", "did", "is", "are", "was", "were", "can",
    "could", "should", "would", "will", "the", "a", "an", "this", "that",
    "these", "those", "in", "from", "inside", "within", "using",
}

QUERY_ACTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"explain|describe|summarise|summarize|summary\s+of|tell\s+me\s+about|"
    r"what\s+happened\s+in|what\s+is\s+in|what\s+does|where\s+does|"
    r"find|show|give\s+me|list"
    r")\s+",
    re.IGNORECASE,
)

QUERY_TRAILING_INTENT_RE = re.compile(
    r"\s+(?:in\s+simple\s+terms|simply|briefly|shortly|with\s+citations|from\s+the\s+text)\s*\??$",
    re.IGNORECASE,
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _slug(value: str, max_len: int = 80) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", _clean(value).lower()).strip("-")
    return clean[:max_len] or "section"


def _normal_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def context_matches(candidate: str, query_context: str) -> bool:
    candidate_terms = {_normal_token(t) for t in re.findall(r"[A-Za-z0-9]+", candidate or "")}
    query_terms = {_normal_token(t) for t in re.findall(r"[A-Za-z0-9]+", query_context or "")}
    candidate_terms.discard("")
    query_terms.discard("")
    return bool(candidate_terms and query_terms and candidate_terms.intersection(query_terms))


def _looks_like_bad_heading_fragment(value: str) -> bool:
    clean = _clean(value).strip()
    lower = clean.lower().strip()

    if not clean:
        return True

    # Reject tiny fragments and common sentence starters.
    if lower in {
        "and", "or", "but", "for", "nor", "so", "yet",
        "the", "a", "an", "this", "that", "these", "those",
        "he", "she", "it", "they", "we", "i", "you",
        "his", "her", "their", "our", "my", "your",
    }:
        return True

    # Reject one-word sentence fragments like "Egypt." or "Zion;".
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", clean)
    if len(words) == 1 and re.search(r"[.;,:!?]$", clean):
        return True

    # Reject very short non-structural headings.
    if len(words) == 1 and len(words[0]) < 4:
        return True

    # Reject lines that look like normal prose, not headings.
    if len(words) > 12:
        return True

    # Reject lines with too much sentence punctuation.
    if len(re.findall(r"[.;!?]", clean)) >= 2:
        return True

    return False


def _looks_like_explicit_heading(value: str) -> bool:
    clean = _clean(value).strip("#*=- ")
    lower = clean.lower()

    patterns = (
        r"^(chapter|section|article|clause|part|appendix|book|page)\s+[a-zA-Z0-9_.:-]+\b",
        r"^the\s+(book|chapter|section|article|part)\s+of\s+",
        r"^appendix\s+[a-zA-Z0-9]+\b",
        r"^[0-9]+(?:\.[0-9]+){0,4}\s+[A-Za-z]",
        r"^[A-Z]\s*\.\s+[A-Za-z]",
    )

    return any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in patterns)


def _looks_like_markdown_heading(raw_line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S+", raw_line or ""))


def _looks_like_title_line(value: str) -> bool:
    clean = _clean(value).strip("#*=- ")
    if _looks_like_bad_heading_fragment(clean):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", clean)
    if not (2 <= len(words) <= 10):
        return False

    lower_words = [w.lower() for w in words]
    weak_starters = {
        "and", "but", "or", "then", "therefore", "because", "when", "where",
        "who", "what", "why", "how", "he", "she", "they", "it", "we",
        "his", "her", "their", "our", "my", "your",
    }
    if lower_words[0] in weak_starters:
        return False

    # Reject sentence/prose fragments. Universal rule:
    # real title lines rarely end with sentence punctuation.
    if re.search(r"[.;,!?]$", clean):
        return False

    # Reject mixed prose after a colon, e.g. "Pharaoh: and Jacob blessed Pharaoh."
    if ":" in clean:
        after_colon = clean.split(":", 1)[1].strip()
        if not after_colon:
            return False
        after_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", after_colon)
        if after_words:
            after_title_ratio = sum(1 for w in after_words if w[:1].isupper() or w.isupper()) / max(len(after_words), 1)
            if after_title_ratio < 0.60:
                return False

    # Good title signal: title case / uppercase-heavy.
    titleish_ratio = sum(1 for w in words if w[:1].isupper() or w.isupper()) / max(len(words), 1)

    if titleish_ratio >= 0.75:
        return True

    return False


def detect_heading(text: str) -> str:
    """
    Universal heading detector.

    Keep real document structure:
      - Markdown headings
      - Chapter/Section/Article/Appendix/Page headings
      - Numbered headings like "2.1 Access Control"
      - Short title-like lines

    Reject prose fragments:
      - "And"
      - "Egypt."
      - "Zion;"
      - random sentence fragments
    """
    raw_lines = [line.rstrip() for line in str(text or "").splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]

    for raw_line in raw_lines[:12]:
        if not raw_line.strip():
            continue

        clean = _clean(raw_line).strip("#*=- ")
        if not clean or len(clean) > 160:
            continue

        if _looks_like_markdown_heading(raw_line):
            return clean

        if _looks_like_explicit_heading(clean):
            return clean

        if _looks_like_title_line(clean):
            return clean

    return ""


def extract_chunk_references(text: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for match in NUMBERED_REF_RE.finditer(text or ""):
        section = str(match.group("section") or "")
        subsection = str(match.group("subsection") or "")
        key = ("numbered_reference", section, subsection)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "reference_type": "numbered_reference",
                "reference_label": f"{section}:{subsection}",
                "section_number": section,
                "subsection_number": subsection,
                "confidence": 0.95,
            }
        )

    for match in EXPLICIT_REF_RE.finditer(text or ""):
        ref_type = str(match.group("type") or "section").lower()
        section = str(match.group("section") or "")
        key = (ref_type, section, "")
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "reference_type": ref_type,
                "reference_label": f"{ref_type.title()} {section}",
                "section_number": section,
                "subsection_number": "",
                "confidence": 0.85,
            }
        )

    return refs[:80]


def _clean_reference_query(query: str) -> str:
    q = QUERY_LEADING_NOISE.sub("", _clean(query))
    q = QUERY_ACTION_PREFIX_RE.sub("", q).strip()
    q = QUERY_TRAILING_INTENT_RE.sub("", q).strip(" ?.,")

    # Remove trailing generic wording after a likely reference:
    # "Psalm 23 in simple terms" -> "Psalm 23"
    q = re.sub(
        r"\s+\b(?:in|with|using|for|about)\b\s+.*$",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip(" ?.,")

    return q


def extract_query_references(query: str) -> list[dict[str, Any]]:
    q = _clean_reference_query(query)
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for match in EXPLICIT_REF_RE.finditer(q):
        ref_type = str(match.group("type") or "section").lower()
        section = str(match.group("section") or "")
        key = (ref_type, section, "")
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "reference_type": ref_type,
                "reference_label": f"{ref_type.title()} {section}",
                "section_number": section,
                "subsection_number": "",
                "named_context": "",
                "confidence": 0.9,
            }
        )

    for match in NAMED_NUMBER_RE.finditer(q):
        context = _clean(match.group("context"))
        section = str(match.group("section") or "")
        subsection = str(match.group("subsection") or "")
        first = context.split()[0].lower() if context else ""

        if first in QUESTION_WORDS or not section:
            continue

        key = ("named_reference", context.lower(), section + ":" + subsection)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "reference_type": "named_reference",
                "reference_label": f"{context} {section}{(':' + subsection) if subsection else ''}",
                "section_number": section,
                "subsection_number": subsection,
                "named_context": context,
                "confidence": 0.88,
            }
        )

    return refs[:6]


def enrich_chunks_with_references(
    chunks: list[dict[str, Any]],
    *,
    source_id: str,
    source_path: str = "",
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    current_heading = ""
    current_parent_id = ""
    total = len(chunks)

    for idx, raw in enumerate(chunks):
        item = dict(raw)
        text = str(item.get("text") or "")

        detected_heading = detect_heading(text)
        if detected_heading:
            current_heading = detected_heading
            digest = sha1(f"{source_id}:{_slug(current_heading)}".encode("utf-8")).hexdigest()[:16]
            current_parent_id = f"parent:{source_id}:{digest}"

        refs = extract_chunk_references(text)

        labels = [r["reference_label"] for r in refs]
        section_numbers = [r["section_number"] for r in refs if r.get("section_number")]
        subsection_numbers = [r["subsection_number"] for r in refs if r.get("subsection_number")]

        item["reference_labels"] = labels
        item["references"] = refs
        item["section_number"] = section_numbers[0] if section_numbers else ""
        item["section_start"] = section_numbers[0] if section_numbers else ""
        item["section_end"] = section_numbers[-1] if section_numbers else ""
        item["subsection_start"] = subsection_numbers[0] if subsection_numbers else ""
        item["subsection_end"] = subsection_numbers[-1] if subsection_numbers else ""
        item["reference_type"] = "numbered_reference_cluster" if len(refs) >= 2 else (refs[0]["reference_type"] if refs else "")
        item["heading"] = detected_heading or current_heading
        item["parent_title"] = current_heading
        item["parent_id"] = current_parent_id
        item["heading_path"] = current_heading
        item["prev_chunk_index"] = idx - 1 if idx > 0 else None
        item["next_chunk_index"] = idx + 1 if idx + 1 < total else None
        item["chunk_strategy"] = item.get("chunk_strategy") or "reference_enriched"
        item["source_path"] = source_path

        enriched.append(item)

    return enriched
