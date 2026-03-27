import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.responses import Response

from app.agent_service import run_agent
from app.audit import agent_run_store
from app.errors import UpstreamAuthError, UpstreamNotFoundError, UpstreamServiceError, UpstreamTimeoutError
from app.embedding_client import generate_embedding
from app.llm_client import generate_reply
from app.observability import get_request_id, metrics_store, reset_request_id, set_request_id
from app.rerank_client import generate_rerank
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentRunsResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)
from app.settings import settings

logger = logging.getLogger("ai_agent.http")

app = FastAPI(title="AI Agent 30D", version="0.7.0")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    token = set_request_id(request_id)
    request.state.request_id = request_id
    started_at = perf_counter()
    response: Response | None = None
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (perf_counter() - started_at) * 1000
        if request.url.path != "/metrics":
            metrics_store.record(request.url.path, status_code, duration_ms)
        logger.info(
            "[%s] %s %s -> %s (%.2f ms)",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        reset_request_id(token)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "day": 7}


@app.get("/metrics")
async def metrics() -> dict[str, object]:
    return {
        "request_id": get_request_id(),
        **metrics_store.snapshot(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = await generate_reply(payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except UpstreamAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail="Chat authentication failed. Check CHAT_OPENAI_API_KEY and CHAT_OPENAI_BASE_URL.",
        ) from exc
    except UpstreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail="Chat request failed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Chat request failed.") from exc

    return ChatResponse(model=settings.chat_model, reply=reply)


@app.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
    try:
        model_name, vector = await generate_embedding(payload.input)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except UpstreamAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail="Embedding authentication failed. Check EMBEDDING_OPENAI_API_KEY and EMBEDDING_OPENAI_BASE_URL.",
        ) from exc
    except UpstreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail="Embedding request failed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Embedding request failed.") from exc

    return EmbeddingResponse(model=model_name, dimensions=len(vector), embedding=vector)


@app.post("/rerank", response_model=RerankResponse)
async def rerank(payload: RerankRequest) -> RerankResponse:
    try:
        model_name, results = await generate_rerank(payload.query, payload.documents, payload.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except UpstreamAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail="Rerank authentication failed. Check RERANK_OPENAI_API_KEY and RERANK_OPENAI_BASE_URL.",
        ) from exc
    except UpstreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail="Rerank request failed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Rerank request failed.") from exc

    return RerankResponse(model=model_name, query=payload.query, results=results)


@app.post("/agent", response_model=AgentResponse)
async def agent(payload: AgentRequest) -> AgentResponse:
    try:
        return await run_agent(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstreamAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except UpstreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UpstreamTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent request failed.") from exc


@app.get("/agent/runs", response_model=AgentRunsResponse)
async def agent_runs(
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> AgentRunsResponse:
    return AgentRunsResponse(runs=agent_run_store.list_runs(limit=limit, session_id=session_id))
