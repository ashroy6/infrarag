# InfraRAG

InfraRAG is a private, local-first Retrieval-Augmented Generation system for DevOps, cloud, Terraform, runbooks, code repositories, PDFs, DOCX files, and internal engineering knowledge bases.

It is designed as a production-style private AI assistant that can ingest engineering documents, retrieve relevant evidence, rerank context, route questions to the right answer pipeline, generate grounded answers with citations, build a knowledge graph for source inspection, stream responses, and keep an audit trail of what was asked and answered.

---

## Index

1. [InfraRAG Overview](#infrarag)
2. [Architecture Diagram](#architecture_diagram)
3. [Screenshots](#screenshots)
4. [What InfraRAG Does](#what-infrarag-does)
5. [Core Use Cases](#core-use-cases)
6. [Current Stack](#current-stack)
7. [Services](#services)
8. [Latest Features Added](#latest-features-added)
   - [1. Rules-First Intent Router](#1-rules-first-intent-router)
   - [2. RAG Orchestrator](#2-rag-orchestrator)
   - [3. Prompt Library](#3-prompt-library)
   - [4. Multiple Answer Pipelines](#4-multiple-answer-pipelines)
   - [5. Streaming Answer UI](#5-streaming-answer-ui)
   - [6. Latency Timer and Progress Bar](#6-latency-timer-and-progress-bar)
   - [7. Local BGE Reranker](#7-local-bge-reranker)
   - [8. Clickable Citations](#8-clickable-citations)
   - [9. Chat History Support](#9-chat-history-support)
   - [10. SQLite Persistence](#10-sqlite-persistence)
   - [11. Audit Logging](#11-audit-logging)
   - [12. Answer Verifier and Recovery Layer](#12-answer-verifier-and-recovery-layer)
   - [13. Admin UI](#13-admin-ui)
   - [14. Graph UI and Knowledge Graph](#14-graph-ui-and-knowledge-graph)
   - [15. Automatic Graph Build on Ingest](#15-automatic-graph-build-on-ingest)
   - [16. DOCX Ingestion](#16-docx-ingestion)
   - [17. Duplicate Upload Handling](#17-duplicate-upload-handling)
   - [18. Cancellation Support](#18-cancellation-support)
   - [19. Prometheus and Grafana Observability](#19-prometheus-and-grafana-observability)
10. [High-Level Request Flow](#high-level-request-flow)
11. [Question Answering Flow](#question-answering-flow)
12. [Ingestion Flow](#ingestion-flow)
13. [Knowledge Graph Flow](#knowledge-graph-flow)
14. [Project Structure](#project-structure)
15. [Important Backend Files](#important-backend-files)
16. [Supported File Types](#supported-file-types)
17. [Ports](#ports)
18. [Main URLs](#main-urls)
19. [How to Start InfraRAG](#how-to-start-infrarag)
20. [Optional: Ingest Example Repo Files](#optional-ingest-example-repo-files)
21. [API Endpoints](#api-endpoints)
22. [Models Used](#models-used)
23. [Key Runtime Parameters](#key-runtime-parameters)
24. [Chunking Strategy](#chunking-strategy)
25. [Storage Model](#storage-model)
26. [Qdrant Collection](#qdrant-collection)
27. [Example Questions](#example-questions)
28. [Monitoring](#monitoring)
29. [Current Strengths](#current-strengths)
30. [Current Limitations](#current-limitations)
31. [Production Improvements To Add Next](#production-improvements-to-add-next)
32. [Why This Project Matters](#why-this-project-matters)
33. [Thanks](#thanks)

---

## Architecture Diagram

![InfraRAG Architecture](architecture_diagram.png)

---

## Screenshots

![InfraRAG Main UI](infrarag_ui_screenshot.png)

![InfraRAG Detailed UI](infrarag_detailed_ui.png)

![InfraRAG Admin Panel](infrarag_admin_panel.png)

![InfraRAG Chunk UI](infrarag_chunk_ui.png)

---

## What InfraRAG Does

InfraRAG turns private engineering knowledge into a searchable local AI assistant.

It can:

- ingest uploaded files, folders, local paths, Git repositories, PDFs, DOCX files, Markdown, text files, code, Terraform, YAML, JSON, CSV, SQL, logs, and runbooks
- parse and chunk documents using structure-aware chunking where possible
- generate embeddings locally using Ollama and `nomic-embed-text`
- store vectors and metadata in Qdrant
- store source metadata, chunks, conversations, audit logs, and graph data in SQLite
- retrieve the most relevant chunks for a user question
- rerank retrieved chunks using a local BGE reranker
- route questions automatically to the correct answer pipeline
- avoid slow LLM planner calls for common factual, follow-up, story, document, repo, and incident questions through rules-first routing
- generate answers using Ollama and `llama3.2:3b`
- stream answers token-by-token in the UI where supported
- show pipeline metadata such as pipeline name, confidence, router reason, verifier status, graph status, and latency
- provide clickable citations linked to source files, chunks, and PDF page numbers where available
- keep chat history in a ChatGPT-style interface
- support request cancellation from the UI
- build a local knowledge graph for ingested sources
- show graph nodes and relationships in a dedicated Graph UI
- expose infrastructure and RAG metrics through Prometheus and Grafana

---

## Core Use Cases

InfraRAG is useful for:

- explaining private repositories
- asking questions about Terraform modules
- understanding AWS, Kubernetes, CI/CD, and DevOps documentation
- summarising uploaded PDFs, DOCX files, and books
- searching internal runbooks
- explaining incidents and troubleshooting steps
- inspecting source/chunk relationships through a graph view
- building a private engineering knowledge assistant for a company
- demonstrating local AI, RAG, observability, and DevOps platform engineering skills

---

## Current Stack

| Layer | Tool |
|---|---|
| Frontend | Static HTML / JavaScript |
| Admin UI | Static HTML / JavaScript |
| Graph UI | Static HTML / JavaScript / Cytoscape.js |
| Backend | FastAPI |
| LLM Runtime | Ollama |
| Chat Model | `llama3.2:3b` |
| Embedding Model | `nomic-embed-text` |
| Vector Database | Qdrant |
| Reranker | Local BGE reranker |
| Persistent Metadata | SQLite |
| Graph Store | SQLite graph tables |
| Reverse Proxy | Nginx |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Runtime | Docker Compose |

---

## Services

InfraRAG runs as a multi-container Docker Compose application.

Main services:

1. `frontend`
2. `backend`
3. `ollama`
4. `qdrant`
5. `nginx`
6. `prometheus`
7. `grafana`

---

## Latest Features Added

These are the latest major InfraRAG upgrades.

### 1. Rules-First Intent Router

InfraRAG now has a rules-first intent router.

It decides whether a user request should go to:

- normal RAG Q&A
- long explanation
- full document summary
- Terraform/repository/code explanation
- incident or runbook troubleshooting

The router now avoids LLM planner calls for many common cases:

- direct factual questions
- "who/what/where/when/which" questions
- CV/profile/document lookups
- story or character questions
- GitHub repo/name/list questions
- follow-ups such as "elaborate the above"
- numbered follow-ups such as "explain point 2"
- incident/debug/error questions
- repo/code/file explanation questions

This reduces routing latency and avoids wasting 30–90 seconds on the LLM planner for simple questions.

---

### 2. RAG Orchestrator

A central RAG orchestrator controls the main answer flow.

It coordinates:

- routing
- follow-up resolution
- retrieval
- reranking
- prompt selection
- chat history
- citations
- answer verification
- recovery logic
- response formatting
- audit logging

This keeps the backend cleaner and makes new pipelines easier to add.

---

### 3. Prompt Library

Prompts are separated into a dedicated prompt library.

This controls:

- normal answer style
- long explanation style
- document summary style
- Terraform/repo/code explanation style
- incident/runbook style
- citation behavior
- `No evidence found` behavior
- denial recovery behavior
- verifier behavior
- formatting rules

The prompt library now includes stricter evidence rules so the model does not falsely say "No evidence found" when citations exist.

---

### 4. Multiple Answer Pipelines

InfraRAG supports multiple answer pipelines instead of one generic RAG flow.

Current pipeline types:

- normal RAG Q&A
- long explanation
- full document/book summary
- Terraform/repo/code explanation
- incident/runbook troubleshooting

This is better than forcing every question through the same prompt.

---

### 5. Streaming Answer UI

The main UI supports streaming answers from `/api/ask-stream`.

Streaming is used for the main answer generation path, so users can see text arriving instead of waiting silently for the full answer.

Some fallback paths, such as verifier correction or recovery, may still appear after the first draft if they run as a second pass. Long-running paths should continue to be improved so every expensive LLM call streams or shows progress.

---

### 6. Latency Timer and Progress Bar

The UI includes:

- live latency timer in seconds
- progress label
- progress percentage
- compact progress bar
- final latency display

This helps show whether the request is routing, retrieving, generating, verifying, recovering, or completed.

---

### 7. Local BGE Reranker

InfraRAG has a local reranker layer.

The flow is:

1. Qdrant retrieves a larger candidate set
2. BGE reranker scores the candidates locally
3. only the best chunks are passed to Ollama

This improves answer quality because the LLM receives stronger context.

---

### 8. Clickable Citations

Answers support clickable citations.

Citations can point back to:

- source file
- source ID
- source path
- chunk index
- page number for PDFs where available
- retrieved evidence text

This is important because private RAG must prove where the answer came from.

---

### 9. Chat History Support

InfraRAG has chat history support.

The UI supports:

- new chat
- previous chats list
- opening old conversations
- deleting conversations
- recent conversation context for follow-up questions

This moves the app closer to a ChatGPT-style private assistant.

---

### 10. SQLite Persistence

InfraRAG uses SQLite for persistent metadata and history.

SQLite stores:

- source metadata
- uploaded file records
- ingestion status
- chunk metadata
- chat conversations
- chat messages
- audit logs
- graph nodes
- graph edges

The database is stored under the backend data volume.

---

### 11. Audit Logging

InfraRAG includes an audit layer.

The audit trail captures:

- user question
- generated answer
- intent
- pipeline used
- sources used
- model used
- timestamp
- latency
- verifier result
- graph context status where available

This is useful for enterprise-style traceability.

---

### 12. Answer Verifier and Recovery Layer

InfraRAG includes an answer verifier.

The verifier checks whether complex answers are supported by retrieved evidence.

The system also has a generic evidence-denial recovery path:

```text
retrieval found citations
→ model says "No evidence found" or "not mentioned"
→ recovery prompt runs against the same context
→ citations are preserved if useful evidence exists
```

This reduces false "No evidence found" answers.

Normal Q&A usually skips verifier for speed, but denial verification/recovery can run when the model denies evidence despite having citations.

---

### 13. Admin UI

The Admin UI is styled to match the main InfraRAG UI.

It supports:

- source list
- source type
- file type
- source status
- chunk count
- source ID
- last ingested timestamp
- view chunks
- re-ingest source
- upload visibility
- dark theme aligned with the main app

---

### 14. Graph UI and Knowledge Graph

InfraRAG now includes a Graph UI as a third main page beside Chat and Admin.

Main pages:

1. Main Chat UI
2. Admin UI
3. Graph UI

The Graph UI is used to inspect and debug the knowledge base structure.

It shows:

- file nodes
- chunk nodes
- chapter nodes
- section nodes
- concept nodes
- service nodes
- resource nodes
- symbol nodes
- contains relationships
- mentions relationships
- defines relationships
- next relationships
- related_to relationships

The Graph UI includes:

- dark/light theme toggle
- source selector
- search
- node-type filters
- relationship filters
- structured view
- dense explorer
- Graph Mode toggle
- interactive zoom/pan/layout
- node details
- citation opening
- source-scoped graph views

Current graph behavior is mainly for observability, trust, debugging, and knowledge-base inspection. Graph-assisted retrieval exists as an optional context expansion path, but the system is still primarily vector RAG.

---

### 15. Automatic Graph Build on Ingest

InfraRAG now builds graph data during ingestion.

When a new or changed file is ingested:

```text
parse file
→ chunk file
→ embed chunks
→ upsert chunks to Qdrant
→ save source/chunk metadata to SQLite
→ build graph nodes and edges
→ save graph rows to SQLite
```

This means the Graph UI should show graph data immediately after ingestion without manually clicking "Rebuild Graph for All Sources".

Duplicate uploads are also handled better. If a duplicate upload is detected, InfraRAG can check the existing source and build graph data for that existing source if graph data is missing or stale.

---

### 16. DOCX Ingestion

InfraRAG now supports Word `.docx` ingestion.

DOCX files can be uploaded from the main UI and ingested into:

- SQLite source metadata
- SQLite chunk metadata
- Qdrant vector store
- SQLite graph tables

DOCX support is useful for CVs, resumes, design documents, runbooks, project notes, and internal business documents.

Legacy `.doc` files are not currently supported unless converted to `.docx`, `.pdf`, or text first.

---

### 17. Duplicate Upload Handling

Upload handling now detects duplicate files by hash.

Expected behavior:

```text
duplicate upload found
→ check existing source_id
→ skip duplicate vector re-ingestion
→ build graph for existing source if graph is missing or stale
→ return duplicate_files + graph_built metadata
```

This avoids creating repeated source entries while still fixing missing graph rows for existing uploads.

---

### 18. Cancellation Support

The UI supports cancelling active generation.

The backend uses a cancel registry to stop long-running requests where possible.

Cancellation is useful for:

- long explanations
- slow verifier runs
- document summary jobs
- accidental large prompts
- wrong source selection

---

### 19. Prometheus and Grafana Observability

InfraRAG includes observability for both infrastructure/API health and RAG behavior.

Prometheus scrapes backend `/metrics`.

Grafana has an InfraRAG Overview dashboard with:

- total requests
- request rate
- backend memory
- backend CPU
- 5xx errors
- endpoint latency
- HTTP status codes
- RAG questions
- RAG pipeline usage
- graph context enabled
- graph chunks added
- no-evidence answers
- verifier verdicts
- Ollama timeouts
- average RAG answer latency by pipeline

This makes the app easier to demonstrate as a production-style AI/DevOps system.

---

## High-Level Request Flow

```text
User
  |
  v
Frontend UI
  |
  v
Nginx
  |
  v
FastAPI Backend
  |
  v
Follow-up Resolver
  |
  v
Rules-First Intent Router
  |
  +--> Optional LLM Planner only when rules are not enough
  |
  v
Streaming RAG Orchestrator
  |
  +--> Query Embedding using Ollama / nomic-embed-text
  |
  +--> Qdrant Candidate Retrieval
  |
  +--> BGE Reranking
  |
  +--> Optional Graph Context Expansion
  |
  +--> Prompt Library
  |
  +--> Ollama / llama3.2:3b
  |
  +--> Optional Verifier / Recovery Layer
  |
  +--> Citation Formatter
  |
  +--> SQLite Audit Log
  |
  v
Frontend Answer with Metadata, Latency, Progress, and Citations
```

---

## Question Answering Flow

```text
Question
  |
  v
Check if follow-up
  |
  v
Resolve standalone question if needed
  |
  v
Choose pipeline
  |
  +--> rules_first when obvious
  |
  +--> llm_planner fallback when ambiguous
  |
  v
Retrieve candidate chunks from Qdrant
  |
  v
Rerank chunks with BGE
  |
  v
Compact context
  |
  v
Build pipeline-specific prompt
  |
  v
Stream answer from Ollama
  |
  v
Optionally verify/recover
  |
  v
Save chat + audit
  |
  v
Show answer + citations
```

---

## Ingestion Flow

```text
Files / Folders / Uploads / PDFs / DOCX / Repos
  |
  v
File Discovery
  |
  v
Parser
  |
  v
Chunker
  |
  v
Embedding Service
  |
  v
Qdrant Vector Store
  |
  v
SQLite Metadata Store
  |
  v
Graph Extractor
  |
  v
SQLite Graph Store
```

---

## Knowledge Graph Flow

```text
Ingested chunks
  |
  v
Graph Extractor
  |
  +--> file nodes
  +--> chunk nodes
  +--> section/chapter nodes
  +--> concept nodes
  +--> service nodes
  +--> resource nodes
  +--> symbol nodes
  |
  v
Graph Store in SQLite
  |
  v
Graph API
  |
  v
Graph UI with Cytoscape.js
```

---

## Project Structure

Current project tree summary:

```text
infrarag/
├── architectural-diagram-infrarag.png
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── router.py
│       ├── rag_orchestrator.py
│       ├── streaming_orchestrator.py
│       ├── prompts.py
│       ├── retrieve.py
│       ├── reranker.py
│       ├── response_formatter.py
│       ├── answer_verifier.py
│       ├── audit.py
│       ├── chat_history.py
│       ├── followup_resolver.py
│       ├── query_planner.py
│       ├── metadata_db.py
│       ├── qdrant_client.py
│       ├── embedding_service.py
│       ├── uploads.py
│       ├── pipeline.py
│       ├── pipelines/
│       ├── ingest.py
│       ├── incremental_ingest.py
│       ├── delete_sync.py
│       ├── git_connector.py
│       ├── pdf_parser.py
│       ├── docx_parser.py
│       ├── text_parser.py
│       ├── text_chunker.py
│       ├── code_parser.py
│       ├── code_chunker.py
│       ├── context_utils.py
│       ├── cancel_registry.py
│       ├── summary_jobs.py
│       ├── graph_api.py
│       ├── graph_extractor.py
│       ├── graph_store.py
│       ├── graph_retrieval.py
│       ├── rag_metrics.py
│       └── state_store.py
├── docs/
│   ├── runbooks/
│   └── uploads/
├── frontend/
│   ├── index.html
│   ├── admin.html
│   └── graph.html
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       └── dashboards/
├── nginx/
│   └── nginx.conf
├── docker-compose.yml
└── README.md
```

---

## Important Backend Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI entrypoint and API routes |
| `router.py` | Rules-first router that decides which answer pipeline to use |
| `query_planner.py` | Optional LLM planner fallback for ambiguous questions |
| `followup_resolver.py` | Resolves vague follow-ups such as "elaborate the above" |
| `rag_orchestrator.py` | Non-streaming RAG answer controller |
| `streaming_orchestrator.py` | Main streaming answer controller used by the UI |
| `prompts.py` | Central prompt library |
| `retrieve.py` | Vector retrieval, clustering, reranking, and graph context expansion |
| `reranker.py` | Local BGE reranking |
| `context_utils.py` | Builds compact context and citations |
| `response_formatter.py` | Formats no-evidence and response objects |
| `answer_verifier.py` | Verifies and repairs unsupported or false-denial answers |
| `audit.py` | Saves audit events |
| `chat_history.py` | Conversation and message handling |
| `metadata_db.py` | SQLite metadata, files, chunks, audit, and history storage |
| `qdrant_client.py` | Qdrant wrapper |
| `embedding_service.py` | Embedding generation |
| `uploads.py` | File and folder upload support |
| `pipeline.py` | Ingestion pipeline |
| `incremental_ingest.py` | Incremental ingestion support |
| `delete_sync.py` | Delete/sync logic for local and Git sources |
| `git_connector.py` | Git repository ingestion support |
| `pdf_parser.py` | PDF parsing |
| `docx_parser.py` | DOCX parsing |
| `text_parser.py` | Text, Markdown, JSON, CSV, config parsing |
| `code_parser.py` | Code parsing |
| `text_chunker.py` | Text chunking |
| `code_chunker.py` | Code-aware chunking |
| `graph_api.py` | Graph API endpoints |
| `graph_extractor.py` | Builds graph nodes and edges from chunks |
| `graph_store.py` | Stores and retrieves graph nodes/edges in SQLite |
| `graph_retrieval.py` | Adds optional graph-related chunks to RAG context |
| `rag_metrics.py` | Custom Prometheus RAG metrics |
| `cancel_registry.py` | Request cancellation support |
| `summary_jobs.py` | Background-style document summary job tracking |

---

## Supported File Types

InfraRAG currently supports these file types for ingestion:

| Type | Extensions |
|---|---|
| PDF | `.pdf` |
| Word document | `.docx` |
| Markdown | `.md` |
| Text | `.txt`, `.rst`, `.log` |
| Config | `.ini`, `.cfg`, `.conf`, `.env` |
| Data/text structured files | `.json`, `.csv` |
| YAML | `.yml`, `.yaml` |
| Terraform / HCL | `.tf`, `.hcl` |
| Shell | `.sh`, `.bash` |
| SQL | `.sql` |
| JavaScript / TypeScript | `.js`, `.ts`, `.tsx`, `.jsx` |
| Python | `.py` |
| Java | `.java` |
| Go | `.go` |
| Rust | `.rs` |
| Dockerfile | `Dockerfile` |

Currently not supported directly:

- legacy `.doc`
- scanned image-only PDFs without OCR
- PowerPoint `.pptx`
- Excel `.xlsx`
- images as knowledge sources
- audio/video transcripts unless converted to text first

---

## Ports

| Service | Port | Purpose |
|---|---:|---|
| Frontend container | 3000 | Raw frontend container |
| Backend API | 8000 | FastAPI backend |
| Nginx app entry | 8080 | Main application URL |
| Ollama host port | 11435 | Host access to Ollama container |
| Qdrant HTTP | 6333 | Qdrant API |
| Qdrant gRPC | 6334 | Qdrant gRPC |
| Prometheus | 9090 | Metrics UI |
| Grafana | 3001 | Dashboard UI |

---

## Main URLs

| URL | Purpose |
|---|---|
| `http://localhost:8080` | Main InfraRAG chat app |
| `http://localhost:8080/admin.html` | Admin UI |
| `http://localhost:8080/graph.html` | Graph UI |
| `http://localhost:9090` | Prometheus |
| `http://localhost:3001` | Grafana |
| `http://localhost:6333` | Qdrant API |

---

## How to Start InfraRAG

```bash
git clone https://github.com/ashroy6/infrarag.git
cd infrarag
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2:3b
```

Open the app:

```text
http://localhost:8080
```

Open the admin page:

```text
http://localhost:8080/admin.html
```

Open the graph page:

```text
http://localhost:8080/graph.html
```

After cloning InfraRAG, each user must ingest their own files or upload documents through the UI. The GitHub repo contains the application code, not a prebuilt private knowledge base.

---

## Optional: Ingest Example Repo Files

```bash
docker compose exec backend python app/ingest.py repos/ml_data_pipeline/src repos/ml_data_pipeline/README.md
```

Use this only if those files exist inside your mounted `docs/` corpus.

---

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Backend status |
| `GET /health` | Health check |
| `GET /retrieve?q=...` | Retrieve chunks only |
| `GET /ask?q=...` | Non-streaming RAG answer flow |
| `GET /ask-stream?q=...` | Streaming RAG answer flow |
| `POST /cancel` | Cancel active request |
| `POST /upload` | Upload and ingest one file |
| `POST /upload-folder` | Upload and ingest folder |
| `POST /ingest/local` | Ingest local path |
| `POST /ingest/git` | Clone/pull and ingest Git repo |
| `GET /citation` | Retrieve exact citation evidence |
| `GET /conversations` | List conversations |
| `GET /conversations/{id}/messages` | Get conversation messages |
| `PATCH /conversations/{id}` | Rename conversation |
| `DELETE /conversations/{id}` | Delete conversation |
| `GET /admin/stats` | Admin stats |
| `GET /admin/sources` | List sources |
| `GET /admin/chunks` | List chunks for a source |
| `POST /admin/reingest` | Re-ingest source |
| `DELETE /admin/source` | Delete source |
| `GET /graph/stats` | Graph stats |
| `GET /graph` | Graph data |
| `GET /graph/source/{source_id}` | Graph data for one source |
| `POST /graph/rebuild-source/{source_id}` | Rebuild graph for one source |
| `POST /graph/rebuild` | Rebuild graph for active sources |
| `GET /audits` | List audit logs |
| `GET /audits/export` | Export audit logs |
| `GET /metrics` | Prometheus metrics |

---

## Models Used

### Embedding Model

```text
nomic-embed-text
```

Used for:

- document embeddings
- query embeddings

### Chat Model

```text
llama3.2:3b
```

Used for:

- answering grounded questions
- long explanations
- document summaries
- repo and Terraform explanations
- verifier/recovery paths
- optional LLM planner fallback

### Reranker

```text
Local BGE reranker
```

Used for:

- reranking retrieved chunks before they are sent to the LLM

---

## Key Runtime Parameters

Important parameters are configured through `docker-compose.yml` environment variables and backend defaults.

| Parameter | Purpose |
|---|---|
| `CHAT_MODEL` | Main Ollama chat model |
| `EMBED_MODEL` | Embedding model |
| `PLANNER_ENABLED` | Enables/disables LLM planner fallback |
| `PLANNER_TIMEOUT_SECONDS` | Max planner wait time |
| `PLANNER_NUM_PREDICT` | Planner output token budget |
| `FOLLOWUP_RESOLVER_ENABLED` | Enables follow-up resolver |
| `FOLLOWUP_RESOLVER_TIMEOUT_SECONDS` | Max follow-up resolver wait |
| `NORMAL_QA_LIMIT` | Final chunks for normal Q&A |
| `NORMAL_QA_NUM_PREDICT` | Normal Q&A answer token budget |
| `LONG_EXPLANATION_LIMIT` | Final chunks for long explanation |
| `LONG_EXPLANATION_NUM_PREDICT` | Long explanation token budget |
| `REPO_EXPLANATION_LIMIT` | Final chunks for repo explanation |
| `REPO_EXPLANATION_NUM_PREDICT` | Repo explanation token budget |
| `CODE_FILE_NUM_PREDICT` | Exact file/code answer token budget |
| `INCIDENT_RUNBOOK_LIMIT` | Final chunks for incident/runbook |
| `INCIDENT_RUNBOOK_NUM_PREDICT` | Incident/runbook token budget |
| `DOC_SUMMARY_BATCH_SIZE` | Batch size for document summary |
| `DOC_SUMMARY_MAP_NUM_PREDICT` | Map-stage summary token budget |
| `DOC_SUMMARY_REDUCE_NUM_PREDICT` | Reduce-stage summary token budget |
| `VERIFY_COMPLEX_ANSWERS` | Enables verifier for complex pipelines |
| `NORMAL_QA_DENIAL_VERIFIER_ENABLED` | Enables verifier only when normal QA denies evidence despite citations |
| `VERIFIER_TIMEOUT_SECONDS` | Max verifier wait |
| `VERIFIER_NUM_PREDICT` | Verifier token budget |
| `MIN_SCORE_THRESHOLD` | Minimum retrieval confidence threshold |
| `MAX_CONTEXT_CHARS_PER_CHUNK` | Per-chunk context cap |
| `MAX_TOTAL_CONTEXT_CHARS` | Total context cap |

---

## Chunking Strategy

InfraRAG supports structure-aware chunking.

### Markdown

Chunked by headings where possible.

### Terraform

Chunked by logical Terraform blocks such as:

- `resource`
- `module`
- `variable`
- `output`
- `data`
- `provider`
- `terraform`

### Python

Chunked by function or class where possible.

### YAML

Chunked by logical top-level sections.

### PDF

Parsed page by page where possible, with page metadata preserved for citations.

### DOCX

Parsed from Word document body text into chunkable text.

### Fallback

If structure-aware chunking is not possible, InfraRAG falls back to fixed-size chunking.

---

## Storage Model

InfraRAG separates raw files, vector data, metadata, graph data, and runtime logs.

| Storage | Purpose |
|---|---|
| `docs/` | Mounted local source knowledge |
| `/app/data/uploads` | Uploaded files inside backend data volume |
| `/app/data/repos` | Cloned Git repositories |
| Qdrant | Vector embeddings and searchable chunks |
| SQLite | Source records, chunk metadata, chat history, audit logs, graph data |
| Docker volumes | Persistent container data |
| Prometheus volume/config | Metrics scraping |
| Grafana volume/config | Dashboards and visualisation |

---

## Qdrant Collection

Default collection:

```text
infrarag_docs
```

Each vector point can include metadata such as:

- source path
- source ID
- source type
- file type
- parser type
- chunk index
- chunk text
- page number where available
- page range where available

---

## Example Questions

After ingestion, you can ask:

### Document and Profile Questions

- What is this document about?
- Summarise this uploaded PDF.
- Summarise this uploaded DOCX.
- What are the key points in this profile?
- What certifications are listed in this document?
- What experience is mentioned in this CV or resume?
- Which projects are mentioned in this profile?
- Which source supports this answer?

### Repository and Code Questions

- What does this repo do?
- Explain this project structure.
- Explain this Python file.
- Explain this Terraform module.
- Which file defines this resource?
- Which file creates this infrastructure component?
- Explain this CI/CD workflow.
- What does this deployment script do?

### DevOps, Cloud, and Runbook Questions

- What are the main steps in this runbook?
- What caused this incident according to the runbook?
- What should I check first for this error?
- What rollback path is described in this document?
- Which AWS services are mentioned in this source?
- What monitoring or observability steps are described?

### Story, Book, and Long Document Questions

- Who is the main character in this story?
- What is the story about?
- Summarise chapter 1.
- Explain the main theme of this document.
- Elaborate the previous answer.
- Explain point 2 in detail.


---

## Monitoring

InfraRAG includes monitoring through Prometheus and Grafana.

### Prometheus

Prometheus scrapes backend metrics from:

```text
/metrics
```

Open Prometheus:

```text
http://localhost:9090
```

### Grafana

Open Grafana:

```text
http://localhost:3001
```

Common first login:

```text
username: admin
password: admin
```

Grafana dashboard coverage includes:

- total requests
- requests per second
- 5xx errors per minute
- backend memory
- backend CPU
- request rate by endpoint
- endpoint p95 latency
- HTTP status codes
- RAG questions
- graph chunks added
- no-evidence answers
- Ollama timeouts
- average RAG answer latency by pipeline
- pipeline usage
- verifier verdicts
- graph context enabled

---

## Current Strengths

InfraRAG now demonstrates:

- private local RAG
- local LLM inference with Ollama
- local embeddings
- Qdrant vector search
- local BGE reranking
- rules-first intent routing
- optional LLM planner fallback
- multiple answer pipelines
- prompt library separation
- streaming answer generation
- request cancellation
- answer verification
- denial recovery
- clickable citations
- chat history
- audit trail
- file and folder upload support
- PDF ingestion
- DOCX ingestion
- Git/repo ingestion support
- incremental ingestion support
- duplicate upload handling
- delete/sync support
- Admin UI
- Graph UI
- SQLite-backed knowledge graph
- automatic graph build on ingest
- optional graph context expansion
- Prometheus metrics
- Grafana dashboard
- Docker Compose deployment

---

## Current Limitations

Current limitations:

- no authentication yet
- no role-based access control yet
- no production secret manager integration yet
- no HTTPS/TLS configuration yet
- no multi-user access isolation yet
- `.doc` legacy Word files are not supported directly
- scanned/image-only PDFs need OCR before ingestion
- PowerPoint and Excel ingestion are not supported yet
- very large corpora need better indexing, batching, and queue management
- some secondary LLM paths such as verifier/recovery may still appear after the first streamed draft
- answer quality gates need more formal tests
- graph-assisted retrieval is still early and optional
- graph is currently mostly for observability/debugging, not the primary retrieval engine
- full CI/CD pipeline can be improved further
- production deployment needs TLS, auth, backup, monitoring alerts, and access control

---

## Production Improvements To Add Next

Recommended next improvements:

1. Add authentication and user login
2. Add role-based access control
3. Add proper HTTPS/TLS
4. Add backup and restore for Qdrant and SQLite
5. Add stronger CI/CD tests
6. Add Docker Compose smoke tests in GitHub Actions
7. Add regression tests for known questions and expected citations
8. Add answer-quality gates for too-short long explanations
9. Add story/book neighboring-chunk retrieval
10. Add better Graph-RAG retrieval using graph relationships
11. Add ingestion queue and background jobs
12. Add source deletion/rebuild from Admin UI
13. Add model and retrieval evaluation tests
14. Add confidence thresholds for `No evidence found`
15. Add source-level access permissions
16. Add deployment guide for VM or cloud server
17. Add production secrets handling
18. Add OCR support for scanned PDFs
19. Add PowerPoint and Excel ingestion
20. Add admin page for re-indexing, graph rebuilds, and corpus health

---

## Why This Project Matters

InfraRAG is not just a chatbot.

It demonstrates how to build a private engineering knowledge assistant over:

- code
- runbooks
- infrastructure documentation
- Terraform modules
- DevOps documentation
- CI/CD pipelines
- PDFs
- DOCX files
- internal repositories
- graph-inspected knowledge bases

It brings together:

- DevOps engineering
- platform engineering
- AI infrastructure
- local LLM operations
- retrieval-augmented generation
- vector databases
- reranking
- knowledge graphs
- observability
- Docker-based deployment
- auditability
- enterprise-style architecture thinking

---

## Thanks

Built by Ashish Dev as a local-first private AI/RAG platform for DevOps, cloud, code, documents, and engineering knowledge.
