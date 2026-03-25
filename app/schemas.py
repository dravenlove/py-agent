from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    model: str
    reply: str


class EmbeddingRequest(BaseModel):
    input: str = Field(min_length=1, max_length=4000)


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embedding: list[float]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)


class RerankResult(BaseModel):
    index: int | None
    score: float | None
    document: str


class RerankResponse(BaseModel):
    model: str
    query: str
    results: list[RerankResult]
