from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.metadata_db import MetadataDB
from app.pipeline import ingest_paths
from app.uploads import save_uploads

router = APIRouter(prefix="/agents", tags=["Agent Studio"])

VALID_DOMAINS = {
    "document_knowledge",
    "customer_support",
    "hr",
    "github",
    "gmail",
    "it_helpdesk",
    "finance",
    "legal",
    "general",
}

VALID_SECURITY_LEVELS = {
    "public",
    "internal",
    "confidential",
    "restricted",
}


def _clean_text(value: str | None, default: str) -> str:
    clean = (value or "").strip()
    return clean or default


def _parse_tags(tags: str | None) -> list[str]:
    if not tags:
        return []

    # Accept either JSON array or comma-separated text.
    try:
        parsed = json.loads(tags)
        if isinstance(parsed, list):
            out: list[str] = []
            for item in parsed:
                value = str(item or "").strip()
                if value and value not in out:
                    out.append(value)
            return out
    except Exception:
        pass

    out = []
    for part in tags.split(","):
        value = part.strip()
        if value and value not in out:
            out.append(value)

    return out


def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}

    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata_json must be valid JSON: {exc}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail="metadata_json must be a JSON object",
        )

    return parsed


def _normalise_domain(data_domain: str | None) -> str:
    domain = _clean_text(data_domain, "general").lower()

    # Friendly aliases for UI/user input.
    aliases = {
        "document": "document_knowledge",
        "documents": "document_knowledge",
        "knowledge": "document_knowledge",
        "customer": "customer_support",
        "support": "customer_support",
        "customer_support_agent": "customer_support",
        "github_agent": "github",
        "gmail_agent": "gmail",
        "email": "gmail",
        "it": "it_helpdesk",
        "helpdesk": "it_helpdesk",
        "it_helpdesk_agent": "it_helpdesk",
        "legal_agent": "legal",
        "finance_agent": "finance",
        "hr_agent": "hr",
    }

    domain = aliases.get(domain, domain)

    if domain not in VALID_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_domain '{data_domain}'. Allowed: {sorted(VALID_DOMAINS)}",
        )

    return domain


def _normalise_security_level(security_level: str | None) -> str:
    level = _clean_text(security_level, "internal").lower()

    if level not in VALID_SECURITY_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid security_level '{security_level}'. Allowed: {sorted(VALID_SECURITY_LEVELS)}",
        )

    return level


@router.get("/health")
def agents_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "component": "agent_studio",
        "features": {
            "knowledge_sources": True,
            "agent_runtime": True,
            "tool_registry": False,
            "approvals": False,
        },
    }


@router.get("/knowledge-sources")
def list_knowledge_sources(
    tenant_id: str = "local",
    data_domain: str | None = None,
    status: str | None = "active",
    limit: int = 200,
) -> dict[str, Any]:
    db = MetadataDB()

    normalised_domain = _normalise_domain(data_domain) if data_domain else None

    sources = db.list_knowledge_sources(
        tenant_id=tenant_id,
        source_group="agent_studio",
        data_domain=normalised_domain,
        status=status,
        limit=limit,
    )

    return {
        "knowledge_sources": sources,
        "count": len(sources),
    }


@router.get("/knowledge-sources/{knowledge_source_id}")
def get_knowledge_source(knowledge_source_id: str) -> dict[str, Any]:
    db = MetadataDB()
    source = db.get_knowledge_source(knowledge_source_id)

    if not source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    files = db.list_active_files(
        source_group="agent_studio",
        knowledge_source_id=knowledge_source_id,
    )

    return {
        "knowledge_source": source,
        "files": files,
        "file_count": len(files),
    }


@router.post("/knowledge-sources/upload")
async def upload_knowledge_source(
    files: list[UploadFile] = File(...),
    source_name: str = Form(...),
    source_description: str | None = Form(None),
    folder_name: str | None = Form(None),
    tenant_id: str = Form("local"),
    owner_user_id: str = Form("ashish"),
    data_domain: str = Form("general"),
    security_level: str = Form("internal"),
    tags: str | None = Form(None),
    metadata_json: str | None = Form(None),
) -> dict[str, Any]:
    """
    Production-style Agent Studio source onboarding.

    One upload/folder creates one parent knowledge source.
    Each file under that upload is linked to the same knowledge_source_id.
    Every new Qdrant chunk receives agent metadata for retrieval filtering.

    Existing regular InfraRAG chat sources are not touched.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    clean_source_name = _clean_text(source_name, "Knowledge Source")
    clean_folder_name = _clean_text(folder_name, clean_source_name)
    clean_tenant_id = _clean_text(tenant_id, "local")
    clean_owner_user_id = _clean_text(owner_user_id, "ashish")
    clean_domain = _normalise_domain(data_domain)
    clean_security = _normalise_security_level(security_level)
    clean_tags = _parse_tags(tags)
    custom_metadata = _parse_metadata(metadata_json)

    db = MetadataDB()

    knowledge_source_id = db.create_knowledge_source(
        source_name=clean_source_name,
        source_description=source_description,
        tenant_id=clean_tenant_id,
        connector="file_upload",
        source_group="agent_studio",
        data_domain=clean_domain,
        security_level=clean_security,
        owner_user_id=clean_owner_user_id,
        tags=clean_tags,
        metadata={
            **custom_metadata,
            "upload_mode": "agent_studio",
            "folder_name": clean_folder_name,
        },
        status="active",
    )

    try:
        base_dir, saved_paths = await save_uploads(
            files=files,
            folder_name=f"agent_studio/{knowledge_source_id}/{clean_folder_name}",
        )

        ingest_result = ingest_paths(
            paths=[base_dir],
            source_type="agent_upload",
            tenant_id=clean_tenant_id,
            owner_user_id=clean_owner_user_id,
            source_group="agent_studio",
            connector="file_upload",
            data_domain=clean_domain,
            security_level=clean_security,
            tags=clean_tags,
            metadata={
                **custom_metadata,
                "knowledge_source_id": knowledge_source_id,
                "source_name": clean_source_name,
                "base_dir": base_dir,
                "saved_paths": saved_paths,
            },
            agent_access_enabled=True,
            knowledge_source_id=knowledge_source_id,
        )

    except Exception as exc:
        db.set_knowledge_source_status(knowledge_source_id, "failed")
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge source upload/ingestion failed: {exc}",
        ) from exc

    files_after_ingest = db.list_active_files(
        source_group="agent_studio",
        knowledge_source_id=knowledge_source_id,
    )

    return {
        "status": "indexed",
        "knowledge_source_id": knowledge_source_id,
        "source_name": clean_source_name,
        "tenant_id": clean_tenant_id,
        "data_domain": clean_domain,
        "security_level": clean_security,
        "agent_access_enabled": True,
        "base_dir": base_dir,
        "saved_file_count": len(saved_paths),
        "indexed_file_count": len(files_after_ingest),
        "files": files_after_ingest,
        "ingest_result": ingest_result,
    }


@router.post("/agents")
def create_agent(
    agent_name: str = Form(...),
    agent_description: str | None = Form(None),
    agent_type: str = Form("knowledge_agent"),
    instructions: str | None = Form(None),
    tenant_id: str = Form("local"),
    created_by: str = Form("ashish"),
    status: str = Form("draft"),
    metadata_json: str | None = Form(None),
) -> dict[str, Any]:
    db = MetadataDB()
    metadata = _parse_metadata(metadata_json)

    agent_id = db.create_agent(
        agent_name=agent_name,
        agent_description=agent_description,
        agent_type=agent_type,
        instructions=instructions,
        tenant_id=tenant_id,
        created_by=created_by,
        metadata=metadata,
        status=status,
    )

    agent = db.get_agent(agent_id)

    return {
        "status": "created",
        "agent_id": agent_id,
        "agent": agent,
    }


@router.get("/agents")
def list_agents(
    tenant_id: str = "local",
    status: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    db = MetadataDB()
    agents = db.list_agents(
        tenant_id=tenant_id,
        status=status,
        limit=limit,
    )

    return {
        "agents": agents,
        "count": len(agents),
    }


@router.post("/agents/{agent_id}/sources/{knowledge_source_id}")
def assign_source_to_agent(
    agent_id: str,
    knowledge_source_id: str,
    tenant_id: str = Form("local"),
    access_level: str = Form("read"),
    status: str = Form("active"),
) -> dict[str, Any]:
    db = MetadataDB()

    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    source = db.get_knowledge_source(knowledge_source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    agent_source_id = db.assign_knowledge_source_to_agent(
        agent_id=agent_id,
        knowledge_source_id=knowledge_source_id,
        tenant_id=tenant_id,
        access_level=access_level,
        status=status,
    )

    allowed_source_ids = db.list_agent_allowed_source_ids(agent_id)

    return {
        "status": "assigned",
        "agent_source_id": agent_source_id,
        "agent_id": agent_id,
        "knowledge_source_id": knowledge_source_id,
        "allowed_source_ids": allowed_source_ids,
        "allowed_source_count": len(allowed_source_ids),
    }


@router.get("/agents/{agent_id}/sources")
def list_sources_for_agent(agent_id: str) -> dict[str, Any]:
    db = MetadataDB()

    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    sources = db.list_agent_sources(agent_id)
    allowed_source_ids = db.list_agent_allowed_source_ids(agent_id)

    return {
        "agent": agent,
        "sources": sources,
        "allowed_source_ids": allowed_source_ids,
        "source_count": len(sources),
        "allowed_source_count": len(allowed_source_ids),
    }


@router.post("/agents/{agent_id}/run")
def run_agent_endpoint(
    agent_id: str,
    user_task: str = Form(...),
    conversation_id: str | None = Form(None),
    tenant_id: str = Form("local"),
    user_id: str = Form("ashish"),
    limit: int = Form(8),
) -> dict[str, Any]:
    from app.agents.agent_runner import run_agent

    try:
        return run_agent(
            agent_id=agent_id,
            user_task=user_task,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc


@router.get("/runs")
def list_agent_runs(
    agent_id: str | None = None,
    tenant_id: str = "local",
    limit: int = 100,
) -> dict[str, Any]:
    db = MetadataDB()
    runs = db.list_agent_runs(
        agent_id=agent_id,
        tenant_id=tenant_id,
        limit=limit,
    )
    return {
        "agent_runs": runs,
        "count": len(runs),
    }


@router.get("/runs/{agent_run_id}/events")
def list_agent_run_events(agent_run_id: str) -> dict[str, Any]:
    db = MetadataDB()
    events = db.list_agent_run_events(agent_run_id)
    return {
        "agent_run_id": agent_run_id,
        "events": events,
        "count": len(events),
    }
