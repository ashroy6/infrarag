from __future__ import annotations

import os
import re
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import NORMAL_QA_PROMPT
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))
NORMAL_QA_LIMIT = int(os.getenv("NORMAL_QA_LIMIT", "6"))
NORMAL_QA_NUM_PREDICT = int(os.getenv("NORMAL_QA_NUM_PREDICT", "500"))

COMPANY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&.,'’() \-]+?\s(?:Corporation|Company|Limited|Ltd|LLP|PLC|Inc|Corp)\b"
    r"(?:\s*\(trading as [^)]+\))?",
    re.IGNORECASE,
)

UK_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
    re.IGNORECASE,
)

FIELD_START_RE = re.compile(
    r"^(registered in|company number|registration number|registered office|office address|company address|address)\s*:",
    re.IGNORECASE,
)

SECTION_HEADING_RE = re.compile(
    r"^[A-Z][A-Za-z0-9&/() \-]{2,70}:?$"
)


def _is_address_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b(address|registered office|office address|company address|postcode|post code|company number|registration number|registered in|where is it registered)\b",
            q,
        )
    )


def _lines_from_extracted_text(text: str) -> list[str]:
    value = text or ""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s*\|\s*", "\n", value)
    value = re.sub(r"[•●▪◦]", " ", value)
    value = re.sub(r"^\s*page\s*$", "", value, flags=re.IGNORECASE | re.MULTILINE)

    lines: list[str] = []

    for raw in value.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if line.lower() in {"p", "a", "pa"}:
            continue

        lines.append(line)

    return lines


def _looks_like_new_company(line: str) -> bool:
    return COMPANY_RE.search(line or "") is not None


def _looks_like_new_field(line: str) -> bool:
    return FIELD_START_RE.search(line or "") is not None


def _looks_like_section_heading(line: str) -> bool:
    if not line:
        return False

    if _looks_like_new_company(line) or _looks_like_new_field(line):
        return False

    if len(line) > 90:
        return False

    if line.endswith(":"):
        return True

    # Generic heading heuristic, not document-specific.
    words = line.split()
    if 1 <= len(words) <= 7 and SECTION_HEADING_RE.match(line):
        return True

    return False


def _clean_address_value(value: str) -> str:
    address = re.sub(r"\s+", " ", value or "").strip(" .,-")

    postcode = UK_POSTCODE_RE.search(address)
    if postcode:
        address = address[:postcode.end()].strip(" .,-")

    return address


def _extract_registered_offices(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Generic deterministic extractor for company registration/address facts.

    No company names, cities, street names, job titles, or document-specific words
    are hardcoded here.
    """
    candidate_chunks = [
        chunk for chunk in chunks
        if re.search(
            r"\b(registered office|company number|registration number|office address|company address|postcode|post code)\b",
            chunk.get("text", "") or "",
            flags=re.IGNORECASE,
        )
    ]

    if not candidate_chunks:
        return None

    all_lines: list[str] = []
    for chunk in candidate_chunks:
        all_lines.extend(_lines_from_extracted_text(chunk.get("text", "")))

    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    i = 0
    while i < len(all_lines):
        line = all_lines[i]

        company_match = COMPANY_RE.search(line)
        if company_match:
            if current and current.get("registered_office"):
                records.append(current)

            current = {
                "company": company_match.group(0).strip(),
                "company_number": "",
                "registered_office": "",
            }
            i += 1
            continue

        if current is None:
            i += 1
            continue

        lower = line.lower()

        if lower.startswith("company number:") or lower.startswith("registration number:"):
            current["company_number"] = line.split(":", 1)[1].strip()
            i += 1
            continue

        if lower.startswith("registered office:") or lower.startswith("office address:") or lower.startswith("company address:") or lower.startswith("address:"):
            address_parts = [line.split(":", 1)[1].strip()]

            j = i + 1
            while j < len(all_lines):
                next_line = all_lines[j]

                if _looks_like_new_company(next_line):
                    break
                if _looks_like_new_field(next_line):
                    break
                if _looks_like_section_heading(next_line):
                    break

                address_parts.append(next_line)

                joined = " ".join(address_parts)
                if UK_POSTCODE_RE.search(joined):
                    break

                j += 1

            current["registered_office"] = _clean_address_value(" ".join(address_parts))
            i = j + 1
            continue

        i += 1

    if current and current.get("registered_office"):
        records.append(current)

    deduped: list[dict[str, str]] = []
    seen = set()

    for record in records:
        company = re.sub(r"\s+", " ", record.get("company", "")).strip()
        number = re.sub(r"\s+", " ", record.get("company_number", "")).strip()
        office = _clean_address_value(record.get("registered_office", ""))

        key = (company.lower(), number.lower(), office.lower())

        if company and office and key not in seen:
            seen.add(key)
            deduped.append(
                {
                    "company": company,
                    "company_number": number,
                    "registered_office": office,
                }
            )

    if not deduped:
        return None

    output_lines: list[str] = []

    if len(deduped) == 1:
        record = deduped[0]
        output_lines.append(f"{record['company']}:")
        if record["company_number"]:
            output_lines.append(f"- Company number: {record['company_number']}")
        output_lines.append(f"- Registered office: {record['registered_office']}")
    else:
        output_lines.append("The document lists these registered office addresses:")

        for index, record in enumerate(deduped, start=1):
            output_lines.append(f"\n{index}. {record['company']}")
            if record["company_number"]:
                output_lines.append(f"   - Company number: {record['company_number']}")
            output_lines.append(f"   - Registered office: {record['registered_office']}")

    return {
        "answer": "\n".join(output_lines).strip(),
        "records": deduped,
        "candidate_chunks": candidate_chunks,
    }


def run(
    question: str,
    chat_context: str = "",
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    chunks = retrieve_context(
        question,
        limit=NORMAL_QA_LIMIT,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    if not chunks:
        return no_evidence_response()

    top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return no_evidence_response()

    compacted = compact_chunks(chunks, max_total_chars=7000)

    if _is_address_question(question):
        extracted = _extract_registered_offices(compacted)
        if extracted:
            relevant_chunks = extracted.get("candidate_chunks") or compacted
            return {
                "answer": extracted["answer"],
                "citations": build_citations(relevant_chunks[:3]),
                "extraction_mode": "deterministic_registered_office",
            }

    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    prompt = NORMAL_QA_PROMPT.format(
        question=question,
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=NORMAL_QA_NUM_PREDICT,
    )

    if answer.strip() == "No evidence found in the knowledge base.":
        return no_evidence_response()

    return {
        "answer": answer,
        "citations": citations,
    }
