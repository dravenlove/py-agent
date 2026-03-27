from typing import Any

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


class AgentRequest(BaseModel):
    input: str = Field(min_length=1, max_length=4000)
    documents: list[str] | None = None
    top_n: int | None = Field(default=None, ge=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class AgentStep(BaseModel):
    name: str
    status: str
    detail: str


class AgentResponse(BaseModel):
    input: str
    selected_tool: str
    planned_tools: list[str]
    steps: list[AgentStep]
    final_answer: str
    session_id: str | None = None
    memory_used: bool = False
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None
