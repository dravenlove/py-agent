from fastapi import FastAPI, HTTPException
import httpx
from openai import AuthenticationError

from app.embedding_client import generate_embedding
from app.llm_client import generate_reply
from app.rerank_client import generate_rerank
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)
from app.settings import settings

app = FastAPI(title="AI Agent 30D", version="0.2.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "day": 2}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = generate_reply(payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Chat authentication failed. Check CHAT_OPENAI_API_KEY and CHAT_OPENAI_BASE_URL.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Chat request failed.") from exc

    return ChatResponse(model=settings.chat_model, reply=reply)


@app.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
    try:
        model_name, vector = generate_embedding(payload.input)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Embedding authentication failed. Check EMBEDDING_OPENAI_API_KEY and EMBEDDING_OPENAI_BASE_URL.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Embedding request failed.") from exc

    return EmbeddingResponse(model=model_name, dimensions=len(vector), embedding=vector)


@app.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest) -> RerankResponse:
    try:
        model_name, results = generate_rerank(payload.query, payload.documents, payload.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Rerank authentication failed. Check RERANK_OPENAI_API_KEY and RERANK_OPENAI_BASE_URL.",
            ) from exc
        raise HTTPException(status_code=502, detail="Rerank request failed.") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Rerank request failed.") from exc

    return RerankResponse(model=model_name, query=payload.query, results=results)
