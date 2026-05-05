from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


AWS_SERVICE_TERMS = {
    "s3": "Amazon S3",
    "iam": "IAM",
    "kms": "AWS KMS",
    "lambda": "AWS Lambda",
    "ec2": "Amazon EC2",
    "eks": "Amazon EKS",
    "ecs": "Amazon ECS",
    "rds": "Amazon RDS",
    "redshift": "Amazon Redshift",
    "cloudwatch": "Amazon CloudWatch",
    "cloudtrail": "AWS CloudTrail",
    "vpc": "Amazon VPC",
    "route53": "Amazon Route 53",
    "sns": "Amazon SNS",
    "sqs": "Amazon SQS",
    "ecr": "Amazon ECR",
    "alb": "Application Load Balancer",
    "elb": "Elastic Load Balancing",
    "secrets manager": "AWS Secrets Manager",
    "dynamodb": "Amazon DynamoDB",
    "opensearch": "Amazon OpenSearch",
}

DEVOPS_TERMS = {
    "terraform": "Terraform",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "prometheus": "Prometheus",
    "grafana": "Grafana",
    "alertmanager": "Alertmanager",
    "ansible": "Ansible",
    "helm": "Helm",
    "ci/cd": "CI/CD",
    "pipeline": "Pipeline",
    "observability": "Observability",
    "monitoring": "Monitoring",
    "logging": "Logging",
    "incident": "Incident",
    "runbook": "Runbook",
    "rollback": "Rollback",
    "deployment": "Deployment",
}

GENERAL_CONCEPT_TERMS = {
    "encryption": "Encryption",
    "authentication": "Authentication",
    "authorization": "Authorization",
    "security": "Security",
    "backup": "Backup",
    "restore": "Restore",
    "latency": "Latency",
    "availability": "Availability",
    "reliability": "Reliability",
    "scalability": "Scalability",
    "networking": "Networking",
    "cost": "Cost",
    "policy": "Policy",
    "access": "Access",
    "compliance": "Compliance",
    "audit": "Audit",
    "governance": "Governance",
    "citation": "Citation",
    "retrieval": "Retrieval",
    "embedding": "Embedding",
    "vector": "Vector Search",
    "rag": "RAG",
    "knowledge graph": "Knowledge Graph",
}

BOOK_CONCEPT_TERMS = {
    "chapter": "Chapter",
    "section": "Section",
    "introduction": "Introduction",
    "summary": "Summary",
    "conclusion": "Conclusion",
    "principle": "Principle",
    "practice": "Practice",
    "method": "Method",
    "argument": "Argument",
    "definition": "Definition",
    "example": "Example",
    "theme": "Theme",
    "character": "Character",
    "philosophy": "Philosophy",
    "yoga": "Yoga",
    "samadhi": "Samadhi",
    "karma": "Karma",
    "atman": "Atman",
    "patanjali": "Patanjali",
}

TERRAFORM_RESOURCE_RE = re.compile(r"\b(aws_[a-z0-9_]+)\b")
PY_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
JS_DEF_RE = re.compile(r"\b(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
PDF_CHAPTER_RE = re.compile(r"\b(chapter|section)\s+([0-9IVXLC]+(?:\.[0-9]+)?)\b[:.\-\s]*(.{0,80})", re.IGNORECASE)


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "item"


def _hash_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _edge_id(source_node_id: str, edge_type: str, target_node_id: str, source_id: str) -> str:
    return "edge:" + _hash_id(source_node_id, edge_type, target_node_id, source_id)


def _node(
    node_id: str,
    node_type: str,
    label: str,
    source_id: str,
    source_path: str,
    chunk_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "label": label,
        "source_id": source_id,
        "chunk_index": chunk_index,
        "source_path": source_path,
        "metadata": metadata or {},
    }


def _edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    source_id: str,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_id": _edge_id(source_node_id, edge_type, target_node_id, source_id),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "edge_type": edge_type,
        "source_id": source_id,
        "weight": weight,
        "metadata": metadata or {},
    }


def _contains_term(text_lower: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def _extract_term_nodes(
    text: str,
    source_id: str,
    source_path: str,
    chunk_index: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """
    Returns nodes and relationships to attach to a chunk.

    tuple is:
      (node_id, edge_type, node_label)
    """
    text_lower = text.lower()
    nodes: list[dict[str, Any]] = []
    mentions: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    service_terms = {**AWS_SERVICE_TERMS}
    concept_terms = {**DEVOPS_TERMS, **GENERAL_CONCEPT_TERMS, **BOOK_CONCEPT_TERMS}

    for term, label in service_terms.items():
        if not _contains_term(text_lower, term):
            continue

        node_id = f"service:{source_id}:{_slug(label)}"
        if node_id not in seen:
            seen.add(node_id)
            nodes.append(
                _node(
                    node_id=node_id,
                    node_type="service",
                    label=label,
                    source_id=source_id,
                    source_path=source_path,
                    metadata={"matched_term": term},
                )
            )
        mentions.append((node_id, "mentions", label))

    for term, label in concept_terms.items():
        if not _contains_term(text_lower, term):
            continue

        node_id = f"concept:{source_id}:{_slug(label)}"
        if node_id not in seen:
            seen.add(node_id)
            nodes.append(
                _node(
                    node_id=node_id,
                    node_type="concept",
                    label=label,
                    source_id=source_id,
                    source_path=source_path,
                    metadata={"matched_term": term},
                )
            )
        mentions.append((node_id, "mentions", label))

    # Keep graph readable.
    return nodes[:18], mentions[:18]


def _extract_resources(
    text: str,
    source_id: str,
    source_path: str,
) -> list[dict[str, Any]]:
    nodes = []
    seen = set()

    for match in TERRAFORM_RESOURCE_RE.findall(text or ""):
        label = match.strip()
        if not label or label in seen:
            continue

        seen.add(label)
        nodes.append(
            _node(
                node_id=f"resource:{source_id}:{_slug(label)}",
                node_type="resource",
                label=label,
                source_id=source_id,
                source_path=source_path,
                metadata={"resource_family": label.split("_", 2)[1] if "_" in label else ""},
            )
        )

    return nodes[:20]


def _extract_code_symbols(
    text: str,
    source_id: str,
    source_path: str,
) -> list[dict[str, Any]]:
    nodes = []
    seen = set()

    for regex in (PY_DEF_RE, JS_DEF_RE):
        for match in regex.findall(text or ""):
            label = match.strip()
            if not label or label in seen:
                continue

            seen.add(label)
            nodes.append(
                _node(
                    node_id=f"symbol:{_slug(Path(source_path).name)}:{_slug(label)}",
                    node_type="symbol",
                    label=label,
                    source_id=source_id,
                    source_path=source_path,
                    metadata={"symbol": label},
                )
            )

    return nodes[:20]


def _extract_section_nodes(
    text: str,
    source_id: str,
    source_path: str,
    chunk_index: int,
) -> list[dict[str, Any]]:
    nodes = []
    seen = set()

    for _, title in HEADING_RE.findall(text or ""):
        clean = re.sub(r"\s+", " ", title).strip(" #:-")
        if not clean or len(clean) > 140:
            continue

        node_id = f"section:{source_id}:{_slug(clean)}"
        if node_id in seen:
            continue

        seen.add(node_id)
        nodes.append(
            _node(
                node_id=node_id,
                node_type="section",
                label=clean[:140],
                source_id=source_id,
                source_path=source_path,
                chunk_index=chunk_index,
                metadata={"detected_from": "markdown_heading"},
            )
        )

    for kind, number, title in PDF_CHAPTER_RE.findall(text or ""):
        clean_title = re.sub(r"\s+", " ", title).strip(" .:-")
        label = f"{kind.title()} {number}"
        if clean_title:
            label = f"{label}: {clean_title[:80]}"

        node_type = "chapter" if kind.lower() == "chapter" else "section"
        node_id = f"{node_type}:{source_id}:{_slug(label)}"

        if node_id in seen:
            continue

        seen.add(node_id)
        nodes.append(
            _node(
                node_id=node_id,
                node_type=node_type,
                label=label[:140],
                source_id=source_id,
                source_path=source_path,
                chunk_index=chunk_index,
                metadata={"detected_from": "text_pattern"},
            )
        )

    return nodes[:8]


def build_graph_for_file(
    source_id: str,
    source_type: str,
    source_path: str,
    file_type: str,
    parser_type: str,
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Build a deterministic graph from one ingested file.

    This v1 extractor intentionally avoids LLM calls.
    Reason: fast ingestion, stable output, no hallucinated graph edges.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    path = Path(source_path)
    file_label = path.name or source_path
    file_node_id = f"file:{source_id}"

    nodes.append(
        _node(
            node_id=file_node_id,
            node_type="file",
            label=file_label,
            source_id=source_id,
            source_path=source_path,
            metadata={
                "source_type": source_type,
                "file_type": file_type,
                "parser_type": parser_type,
                "full_path": source_path,
                "chunk_count": len(chunks),
            },
        )
    )

    previous_chunk_node_id: str | None = None
    previous_section_node_id: str | None = None

    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        page_number = chunk.get("page_number")
        chunk_label = f"Chunk {chunk_index}"
        if page_number is not None:
            chunk_label = f"Page {page_number} / Chunk {chunk_index}"

        preview = re.sub(r"\s+", " ", text[:180]).strip()

        chunk_node_id = f"chunk:{source_id}:{chunk_index}"

        nodes.append(
            _node(
                node_id=chunk_node_id,
                node_type="chunk",
                label=chunk_label,
                source_id=source_id,
                source_path=source_path,
                chunk_index=chunk_index,
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_id": chunk.get("chunk_id"),
                    "qdrant_point_id": chunk.get("qdrant_point_id"),
                    "page_number": page_number,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "preview": preview,
                },
            )
        )

        edges.append(
            _edge(
                source_node_id=file_node_id,
                target_node_id=chunk_node_id,
                edge_type="contains",
                source_id=source_id,
                weight=1.0,
            )
        )

        if previous_chunk_node_id:
            edges.append(
                _edge(
                    source_node_id=previous_chunk_node_id,
                    target_node_id=chunk_node_id,
                    edge_type="next",
                    source_id=source_id,
                    weight=0.4,
                )
            )
        previous_chunk_node_id = chunk_node_id

        section_nodes = _extract_section_nodes(
            text=text,
            source_id=source_id,
            source_path=source_path,
            chunk_index=chunk_index,
        )

        for section_node in section_nodes:
            nodes.append(section_node)
            edges.append(
                _edge(
                    source_node_id=file_node_id,
                    target_node_id=section_node["node_id"],
                    edge_type="contains",
                    source_id=source_id,
                    weight=0.9,
                )
            )
            edges.append(
                _edge(
                    source_node_id=section_node["node_id"],
                    target_node_id=chunk_node_id,
                    edge_type="contains",
                    source_id=source_id,
                    weight=0.9,
                )
            )

            if previous_section_node_id and previous_section_node_id != section_node["node_id"]:
                edges.append(
                    _edge(
                        source_node_id=previous_section_node_id,
                        target_node_id=section_node["node_id"],
                        edge_type="next",
                        source_id=source_id,
                        weight=0.3,
                    )
                )
            previous_section_node_id = section_node["node_id"]

        term_nodes, mentions = _extract_term_nodes(
            text=text,
            source_id=source_id,
            source_path=source_path,
            chunk_index=chunk_index,
        )

        for term_node in term_nodes:
            nodes.append(term_node)

        mentioned_node_ids = []
        for node_id, edge_type, _ in mentions:
            mentioned_node_ids.append(node_id)
            edges.append(
                _edge(
                    source_node_id=chunk_node_id,
                    target_node_id=node_id,
                    edge_type=edge_type,
                    source_id=source_id,
                    weight=0.8,
                )
            )

        resources = _extract_resources(
            text=text,
            source_id=source_id,
            source_path=source_path,
        )

        for resource_node in resources:
            nodes.append(resource_node)
            edges.append(
                _edge(
                    source_node_id=chunk_node_id,
                    target_node_id=resource_node["node_id"],
                    edge_type="defines",
                    source_id=source_id,
                    weight=0.95,
                )
            )

        symbols = _extract_code_symbols(
            text=text,
            source_id=source_id,
            source_path=source_path,
        )

        for symbol_node in symbols:
            nodes.append(symbol_node)
            edges.append(
                _edge(
                    source_node_id=chunk_node_id,
                    target_node_id=symbol_node["node_id"],
                    edge_type="defines",
                    source_id=source_id,
                    weight=0.7,
                )
            )

        # Related concept/service/resource nodes found in the same chunk.
        co_nodes = mentioned_node_ids[:8] + [node["node_id"] for node in resources[:6]]
        co_nodes = list(dict.fromkeys(co_nodes))

        for i, left in enumerate(co_nodes):
            for right in co_nodes[i + 1 : i + 4]:
                edges.append(
                    _edge(
                        source_node_id=left,
                        target_node_id=right,
                        edge_type="related_to",
                        source_id=source_id,
                        weight=0.35,
                        metadata={"reason": "co_occurs_in_chunk", "chunk_index": chunk_index},
                    )
                )

    return {
        "nodes": nodes,
        "edges": edges,
    }
