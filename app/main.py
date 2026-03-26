from fastapi import FastAPI, HTTPException

from app.errors import UpstreamAuthError, UpstreamNotFoundError, UpstreamServiceError, UpstreamTimeoutError
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

app = FastAPI(title="AI Agent 30D", version="0.3.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "day": 3}


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
