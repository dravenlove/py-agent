import pytest

import app.agent_service as agent_service
import app.tools as tools_module
from app.schemas import AgentRequest
from app.tools import ToolExecutionResult


@pytest.mark.anyio
async def test_run_agent_selects_embed_tool(monkeypatch) -> None:
    async def mock_run_embed_text(input_text: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            summary="Embedded text.",
            output={"model": "mock-emb", "dimensions": 3, "embedding_head": [0.1, 0.2, 0.3]},
        )

    monkeypatch.setattr(tools_module, "run_embed_text", mock_run_embed_text)

    response = await agent_service.run_agent(AgentRequest(input="请把这句话转成向量"))

    assert response.selected_tool == "embed_text"
    assert response.steps[1].name == "select_tool"
    assert response.tool_output["dimensions"] == 3


@pytest.mark.anyio
async def test_run_agent_selects_rerank_tool(monkeypatch) -> None:
    async def mock_run_rerank_documents(query: str, documents: list[str], top_n: int | None = None) -> ToolExecutionResult:
        return ToolExecutionResult(
            summary="Reranked docs.",
            output={
                "model": "mock-rerank",
                "results": [{"index": 1, "score": 0.9, "document": documents[1]}],
                "best_match": {"index": 1, "score": 0.9, "document": documents[1]},
            },
        )

    monkeypatch.setattr(tools_module, "run_rerank_documents", mock_run_rerank_documents)

    response = await agent_service.run_agent(
        AgentRequest(input="帮我找出最相关的文档", documents=["文档 1", "文档 2"], top_n=1)
    )

    assert response.selected_tool == "rerank_documents"
    assert response.tool_input["top_n"] == 1
    assert "最相关" in response.final_answer


@pytest.mark.anyio
async def test_run_agent_selects_calculator_tool() -> None:
    response = await agent_service.run_agent(AgentRequest(input="请计算 23 * 7"))

    assert response.selected_tool == "calculator"
    assert response.tool_output["result"] == 161
    assert response.final_answer == "计算完成：23 * 7 = 161"


@pytest.mark.anyio
async def test_run_agent_returns_capability_summary_when_no_tool_matches() -> None:
    response = await agent_service.run_agent(AgentRequest(input="帮我写一首诗"))

    assert response.selected_tool == "unsupported"
    assert "只支持这些工具" in response.final_answer
    assert "available_tools" in response.tool_output
