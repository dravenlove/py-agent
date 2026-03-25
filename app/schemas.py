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
