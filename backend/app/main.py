import os

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.retrieve import retrieve_context

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

app = FastAPI(title="InfraRAG Backend", version="1.0.0")

REQUEST_COUNT = Counter("infrarag_requests_total", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("infrarag_request_latency_seconds", "Request latency", ["endpoint"])


@app.middleware("http")
async def metrics_middleware(request, call_next):
    method = request.method
    endpoint = request.url.path

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)

    return response


def ask_ollama(question: str, context_chunks: list[dict]):
    if not context_chunks:
        return {
            "answer": "No evidence found in the knowledge base.",
            "citations": []
        }

    context_text = "\n\n".join(
        [
            f"[Source: {c['source']} | Chunk: {c['chunk_index']}]\n{c['text']}"
            for c in context_chunks
        ]
    )

    prompt = f"""
You are InfraRAG, a private DevOps and cloud assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say: "No evidence found in the knowledge base."
Be concise and factual.

Context:
{context_text}

Question:
{question}
""".strip()

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        data = response.json()

        citations = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": c["score"]
            }
            for c in context_chunks
        ]

        return {
            "answer": data.get("response", "").strip(),
            "citations": citations
        }

    except Exception as e:
        return {
            "answer": f"Backend/Ollama error: {str(e)}",
            "citations": []
        }


@app.get("/")
def root():
    return {"message": "InfraRAG backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/retrieve")
def retrieve(q: str = ""):
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "query parameter 'q' is required"}
        )

    try:
        hits = retrieve_context(q, limit=5)
        return {"question": q, "results": hits}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"retrieve failed: {str(e)}"}
        )


@app.get("/ask")
def ask(q: str = ""):
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "query parameter 'q' is required"}
        )

    try:
        context_chunks = retrieve_context(q, limit=5)
        result = ask_ollama(q, context_chunks)

        return {
            "question": q,
            "answer": result["answer"],
            "citations": result["citations"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"ask failed: {str(e)}"}
        )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
