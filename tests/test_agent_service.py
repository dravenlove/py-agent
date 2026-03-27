import pytest

import app.agent_service as agent_service
import app.tools as tools_module
from app.memory import memory_store
from app.schemas import AgentRequest
from app.tools import ToolExecutionResult


@pytest.fixture(autouse=True)
def reset_memory() -> None:
    memory_store.reset()


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


@pytest.mark.anyio
async def test_run_agent_plans_rerank_then_summary(monkeypatch) -> None:
    async def mock_run_rerank_documents(query: str, documents: list[str], top_n: int | None = None) -> ToolExecutionResult:
        return ToolExecutionResult(
            summary="Reranked docs.",
            output={
                "model": "mock-rerank",
                "results": [{"index": 0, "score": 0.95, "document": documents[0]}],
                "best_match": {"index": 0, "score": 0.95, "document": documents[0]},
            },
        )

    async def mock_run_summarize_text(text: str) -> ToolExecutionResult:
        return ToolExecutionResult(summary="Summarized text.", output={"source_text": text, "summary": "这是摘要。"})

    monkeypatch.setattr(tools_module, "run_rerank_documents", mock_run_rerank_documents)
    monkeypatch.setattr(tools_module, "run_summarize_text", mock_run_summarize_text)

    response = await agent_service.run_agent(
        AgentRequest(
            input="请先找出最相关的文档，再总结一下",
            documents=["第一段文档", "第二段文档"],
            top_n=1,
        )
    )

    assert response.planned_tools == ["rerank_documents", "summarize_text"]
    assert response.tool_output["summary"] == "这是摘要。"
    assert "总结结果" in response.final_answer


@pytest.mark.anyio
async def test_run_agent_reuses_session_documents() -> None:
    async def mock_run_rerank_documents(query: str, documents: list[str], top_n: int | None = None) -> ToolExecutionResult:
        return ToolExecutionResult(
            summary="Reranked docs.",
            output={
                "model": "mock-rerank",
                "results": [{"index": 0, "score": 0.88, "document": documents[0]}],
                "best_match": {"index": 0, "score": 0.88, "document": documents[0]},
            },
        )

    async def mock_run_summarize_text(text: str) -> ToolExecutionResult:
        return ToolExecutionResult(summary="Summarized text.", output={"source_text": text, "summary": "这是记忆摘要。"})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(tools_module, "run_rerank_documents", mock_run_rerank_documents)
    monkeypatch.setattr(tools_module, "run_summarize_text", mock_run_summarize_text)

    first = await agent_service.run_agent(
        AgentRequest(
            input="帮我找出最相关的文档",
            documents=["文档A", "文档B"],
            session_id="session-1",
        )
    )
    second = await agent_service.run_agent(
        AgentRequest(
            input="继续比较这些文档并总结一下",
            session_id="session-1",
        )
    )

    assert first.session_id == "session-1"
    assert second.memory_used is True
    assert second.tool_input["documents"] == ["文档A", "文档B"]
    assert second.planned_tools[0] == "rerank_documents"
    monkeypatch.undo()


@pytest.mark.anyio
async def test_run_agent_requires_confirmation_before_clearing_memory() -> None:
    memory_store.append_interaction(
        session_id="session-risk",
        user_input="旧请求",
        planned_tools=["rerank_documents"],
        tool_input={"documents": ["文档A"]},
        tool_output={"best_match": {"document": "文档A"}},
        final_answer="旧回答",
    )

    response = await agent_service.run_agent(
        AgentRequest(input="请清空这个会话的记忆", session_id="session-risk")
    )

    assert response.status == "needs_confirmation"
    assert response.selected_tool == "clear_session_memory"
    assert response.approval_required is True
    assert "confirm=true" in response.final_answer
    assert memory_store.get_recent("session-risk")


@pytest.mark.anyio
async def test_run_agent_clears_session_memory_after_confirmation() -> None:
    memory_store.append_interaction(
        session_id="session-risk",
        user_input="旧请求",
        planned_tools=["rerank_documents"],
        tool_input={"documents": ["文档A"]},
        tool_output={"best_match": {"document": "文档A"}},
        final_answer="旧回答",
    )

    response = await agent_service.run_agent(
        AgentRequest(input="请清空这个会话的记忆", session_id="session-risk", confirm=True)
    )

    assert response.status == "completed"
    assert response.selected_tool == "clear_session_memory"
    assert response.tool_output["deleted_count"] == 1
    assert memory_store.get_recent("session-risk") == []


@pytest.mark.anyio
async def test_run_agent_can_reject_risky_action() -> None:
    memory_store.append_interaction(
        session_id="session-risk",
        user_input="旧请求",
        planned_tools=["rerank_documents"],
        tool_input={"documents": ["文档A"]},
        tool_output={"best_match": {"document": "文档A"}},
        final_answer="旧回答",
    )

    response = await agent_service.run_agent(
        AgentRequest(input="请清空这个会话的记忆", session_id="session-risk", confirm=False)
    )

    assert response.status == "cancelled"
    assert response.selected_tool == "clear_session_memory"
    assert memory_store.get_recent("session-risk")
