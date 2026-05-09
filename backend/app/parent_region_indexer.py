from __future__ import annotations

import re
from hashlib import sha1
from typing import Any


ORDINAL_WORDS = {
    "first": "First",
    "second": "Second",
    "third": "Third",
    "fourth": "Fourth",
    "fifth": "Fifth",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normal_key(value: str) -> str:
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", str(value or "").lower()):
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _region_id(source_id: str, title: str, start_idx: int) -> str:
    digest = sha1(f"{source_id}:{_normal_key(title)}:{start_idx}".encode("utf-8")).hexdigest()[:16]
    return f"parent:{source_id}:{digest}"


def _looks_like_global_document_title(value: str) -> bool:
    key = _normal_key(value)
    if not key:
        return True

    global_markers = (
        "title",
        "ebook",
        "project gutenberg",
        "king jame version",
        "release date",
        "language",
        "credit",
        "license",
        "copyright",
    )

    return any(marker in key for marker in global_markers)


def _reference_sections(chunk: dict[str, Any]) -> list[int]:
    values: list[int] = []

    for ref in chunk.get("references", []) or []:
        raw = str(ref.get("section_number") or "").strip()
        if raw.isdigit():
            values.append(int(raw))

    return values


def _has_references(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("references") or chunk.get("reference_labels"))


def _extract_toc_region_names(chunks: list[dict[str, Any]], max_chunks: int = 3) -> list[str]:
    """
    Generic TOC/title extraction.

    Handles patterns such as:
      - The Book of X
      - The First Book of X
      - The First Book of Moses: Called Genesis
      - Book of X

    This is intentionally not tied to any specific source.
    """
    text = "\n".join(str(c.get("text") or "") for c in chunks[:max_chunks])
    text = re.sub(r"\s+", " ", text)

    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        clean = _clean(name).strip(" .,:;\"'")
        clean = re.sub(r"\b(the|book|called)\b", " ", clean, flags=re.IGNORECASE)
        clean = _clean(clean)

        if not clean:
            return
        if len(clean) > 80:
            return
        if len(clean.split()) > 6:
            return

        key = _normal_key(clean)
        if not key or key in seen:
            return

        generic_noise = {
            "project gutenberg",
            "ebook",
            "king james version",
            "title",
            "release date",
            "language",
            "credits",
            "start",
        }
        if key in generic_noise:
            return

        seen.add(key)
        names.append(clean)

    # Prefer explicit "Called X" names where available.
    for match in re.finditer(
        r"\bBook\s+of\s+[^:]{1,80}:\s*Called\s+([A-Z][A-Za-z0-9 _-]{2,60})",
        text,
        flags=re.IGNORECASE,
    ):
        add(match.group(1))

    # Generic "The First Book of X" / "The Book of X".
    for match in re.finditer(
        r"\b(?:The\s+)?(?:(First|Second|Third|Fourth|Fifth)\s+)?Book\s+of\s+([A-Z][A-Za-z0-9 _-]{2,60})",
        text,
        flags=re.IGNORECASE,
    ):
        # Skip "Book of X: Called Y" because the called name was already captured.
        tail = text[match.end(): match.end() + 20]
        if re.search(r"^\s*:\s*Called\b", tail, flags=re.IGNORECASE):
            continue

        ordinal = match.group(1) or ""
        name = match.group(2) or ""
        if ordinal:
            name = f"{ORDINAL_WORDS.get(ordinal.lower(), ordinal)} {name}"
        add(name)

    # Numbered/lettered TOC lines: "1. Introduction", "A. Appendix".
    for match in re.finditer(
        r"(?:^|\s)(?:[0-9]{1,3}|[A-Z])[\).]\s+([A-Z][A-Za-z0-9 _-]{3,70})(?=\s+(?:[0-9]{1,3}|[A-Z])[\).]\s+|$)",
        text,
        flags=re.IGNORECASE,
    ):
        add(match.group(1))

    return names[:300]


def _detect_reference_regions(chunks: list[dict[str, Any]]) -> list[dict[str, int]]:
    """
    Detect parent regions from numbered-reference resets.

    Generic rule:
      If references go from a high section number back to 1/2/3,
      a new parent region likely started.

    Works for books, manuals, policies, logs with repeated local numbering.
    """
    starts: list[int] = []
    previous_max: int | None = None
    previous_idx: int | None = None

    for idx, chunk in enumerate(chunks):
        sections = _reference_sections(chunk)
        if not sections:
            continue

        current_min = min(sections)
        current_max = max(sections)

        if not starts:
            starts.append(idx)
        elif previous_max is not None:
            reset_to_start = current_min <= 2 and previous_max >= 8
            large_drop = current_max + 5 < previous_max and current_min <= 5

            if reset_to_start or large_drop:
                if previous_idx is None or idx > previous_idx:
                    starts.append(idx)

        previous_max = current_max
        previous_idx = idx

    if not starts:
        return []

    regions: list[dict[str, int]] = []

    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else len(chunks) - 1
        if end < start:
            continue
        regions.append({"start_chunk_index": start, "end_chunk_index": end})

    return regions


def _fallback_heading_regions(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        title = _clean(chunk.get("heading") or chunk.get("parent_title") or "")
        if not title:
            continue

        if regions and regions[-1]["parent_title"] == title:
            regions[-1]["end_chunk_index"] = idx
            continue

        if regions:
            regions[-1]["end_chunk_index"] = idx - 1

        regions.append(
            {
                "parent_title": title,
                "parent_type": "heading_region",
                "start_chunk_index": idx,
                "end_chunk_index": idx,
                "confidence": 0.75,
            }
        )

    if regions:
        regions[-1]["end_chunk_index"] = len(chunks) - 1

    return regions


def build_parent_regions_and_enrich_chunks(
    *,
    source_id: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build parent regions and assign chunks to them.

    Returns:
      {
        "chunks": enriched_chunks,
        "parent_regions": [...]
      }
    """
    enriched = [dict(c) for c in chunks]
    toc_names = _extract_toc_region_names(enriched)
    numeric_regions = _detect_reference_regions(enriched)

    parent_regions: list[dict[str, Any]] = []

    if numeric_regions:
        use_toc_names = len(toc_names) >= max(3, min(len(numeric_regions), 10))

        for i, region in enumerate(numeric_regions):
            title = toc_names[i] if use_toc_names and i < len(toc_names) else ""
            if not title:
                candidate_heading = _clean(enriched[region["start_chunk_index"]].get("heading") or "")
                if candidate_heading and not _looks_like_global_document_title(candidate_heading):
                    title = candidate_heading

            if not title:
                title = f"Reference Region {i + 1}"

            toc_name_used = bool(i < len(toc_names))

            parent_regions.append(
                {
                    "parent_id": _region_id(source_id, title, region["start_chunk_index"]),
                    "source_id": source_id,
                    "parent_type": "reference_region",
                    "parent_title": title,
                    "parent_key": _normal_key(title),
                    "start_chunk_index": int(region["start_chunk_index"]),
                    "end_chunk_index": int(region["end_chunk_index"]),
                    "confidence": 0.82 if toc_name_used else 0.55,
                    "metadata": {
                        "region_index": i,
                        "toc_name_used": toc_name_used,
                        "toc_names_available": len(toc_names),
                        "detector": "reference_number_reset",
                    },
                }
            )
    else:
        for region in _fallback_heading_regions(enriched):
            title = region["parent_title"]
            parent_regions.append(
                {
                    "parent_id": _region_id(source_id, title, region["start_chunk_index"]),
                    "source_id": source_id,
                    "parent_type": region["parent_type"],
                    "parent_title": title,
                    "parent_key": _normal_key(title),
                    "start_chunk_index": int(region["start_chunk_index"]),
                    "end_chunk_index": int(region["end_chunk_index"]),
                    "confidence": float(region["confidence"]),
                    "metadata": {
                        "detector": "heading_region",
                    },
                }
            )

    # Assign chunks to parent regions.
    for region in parent_regions:
        start = int(region["start_chunk_index"])
        end = int(region["end_chunk_index"])

        for idx in range(max(0, start), min(len(enriched), end + 1)):
            chunk = enriched[idx]

            if not _has_references(chunk) and not chunk.get("heading"):
                continue

            chunk["parent_id"] = region["parent_id"]
            chunk["parent_title"] = region["parent_title"]
            chunk["parent_region_type"] = region["parent_type"]
            chunk["parent_region_key"] = region["parent_key"]
            chunk["parent_region_confidence"] = region["confidence"]

            if not chunk.get("heading") or str(chunk.get("heading", "")).lower().startswith("title:"):
                chunk["heading"] = region["parent_title"]

            chunk["heading_path"] = region["parent_title"]

    return {
        "chunks": enriched,
        "parent_regions": parent_regions,
    }
