import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.audit import agent_run_store
from app.errors import UpstreamAuthError, UpstreamNotFoundError, UpstreamServiceError, UpstreamTimeoutError
from app.main import app
from app.memory import memory_store
from app.observability import metrics_store
from app.schemas import AgentResponse, AgentStep

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    metrics_store.reset()
    memory_store.reset()
    agent_run_store.reset()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "day": 7}
    assert "X-Request-ID" in response.headers


def test_chat_with_mocked_llm(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        return "mocked answer"

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "mocked answer"
    assert "model" in body


def test_chat_validation() -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_maps_auth_error(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        raise UpstreamAuthError("auth failed")

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_maps_not_found_error(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        raise UpstreamNotFoundError("not found")

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 404


def test_chat_maps_timeout_error(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        raise UpstreamTimeoutError("timed out")

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 504


def test_chat_maps_generic_upstream_error(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        raise UpstreamServiceError("provider failed")

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 502


def test_embeddings_with_mocked_client(monkeypatch) -> None:
    async def mock_generate_embedding(_: str) -> tuple[str, list[float]]:
        return "mock-emb-model", [0.1, 0.2, 0.3]

    monkeypatch.setattr(main_module, "generate_embedding", mock_generate_embedding)
    response = client.post("/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock-emb-model"
    assert body["dimensions"] == 3
    assert body["embedding"] == [0.1, 0.2, 0.3]


def test_embeddings_validation() -> None:
    response = client.post("/embeddings", json={"input": ""})
    assert response.status_code == 422


def test_embeddings_maps_timeout_error(monkeypatch) -> None:
    async def mock_generate_embedding(_: str) -> tuple[str, list[float]]:
        raise UpstreamTimeoutError("timed out")

    monkeypatch.setattr(main_module, "generate_embedding", mock_generate_embedding)
    response = client.post("/embeddings", json={"input": "hello"})
    assert response.status_code == 504


def test_embeddings_maps_auth_error(monkeypatch) -> None:
    async def mock_generate_embedding(_: str) -> tuple[str, list[float]]:
        raise UpstreamAuthError("auth failed")

    monkeypatch.setattr(main_module, "generate_embedding", mock_generate_embedding)
    response = client.post("/embeddings", json={"input": "hello"})
    assert response.status_code == 401


def test_rerank_with_mocked_client(monkeypatch) -> None:
    async def mock_generate_rerank(query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[dict[str, object]]]:
        return "mock-rerank-model", [{"index": 0, "score": 0.97, "document": documents[0]}]

    monkeypatch.setattr(main_module, "generate_rerank", mock_generate_rerank)
    response = client.post(
        "/rerank",
        json={
            "query": "怎么重置密码",
            "documents": ["进入设置页点击重置密码", "查看账单与发票"],
            "top_n": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock-rerank-model"
    assert body["query"] == "怎么重置密码"
    assert body["results"] == [{"index": 0, "score": 0.97, "document": "进入设置页点击重置密码"}]


def test_rerank_maps_auth_error(monkeypatch) -> None:
    async def mock_generate_rerank(query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[dict[str, object]]]:
        raise UpstreamAuthError("auth failed")

    monkeypatch.setattr(main_module, "generate_rerank", mock_generate_rerank)
    response = client.post("/rerank", json={"query": "hello", "documents": ["doc"]})
    assert response.status_code == 401


def test_rerank_maps_timeout_error(monkeypatch) -> None:
    async def mock_generate_rerank(query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[dict[str, object]]]:
        raise UpstreamTimeoutError("timed out")

    monkeypatch.setattr(main_module, "generate_rerank", mock_generate_rerank)
    response = client.post("/rerank", json={"query": "hello", "documents": ["doc"]})
    assert response.status_code == 504


def test_rerank_validation() -> None:
    response = client.post("/rerank", json={"query": "", "documents": []})
    assert response.status_code == 422


def test_metrics_counts_successful_requests(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        return "ok"

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)

    health_response = client.get("/health")
    chat_response = client.post("/chat", json={"message": "hello"})
    metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert chat_response.status_code == 200
    assert metrics_response.status_code == 200

    body = metrics_response.json()
    assert body["requests_total"] == 2
    assert body["failures_total"] == 0
    assert body["routes"]["/health"]["requests"] == 1
    assert body["routes"]["/chat"]["requests"] == 1
    assert body["routes"]["/chat"]["last_status_code"] == 200
    assert body["request_id"] == metrics_response.headers["X-Request-ID"]


def test_metrics_counts_failed_requests(monkeypatch) -> None:
    async def mock_generate_reply(_: str) -> str:
        raise UpstreamTimeoutError("timed out")

    monkeypatch.setattr(main_module, "generate_reply", mock_generate_reply)

    failed_response = client.post("/chat", json={"message": "hello"})
    metrics_response = client.get("/metrics")

    assert failed_response.status_code == 504
    assert metrics_response.status_code == 200

    body = metrics_response.json()
    assert body["requests_total"] == 1
    assert body["failures_total"] == 1
    assert body["routes"]["/chat"]["requests"] == 1
    assert body["routes"]["/chat"]["failures"] == 1
    assert body["routes"]["/chat"]["last_status_code"] == 504


def test_agent_endpoint_with_mocked_service(monkeypatch) -> None:
    async def mock_run_agent(_payload) -> AgentResponse:
        return AgentResponse(
            status="completed",
            run_id="run-123",
            input="请把这句话转成向量",
            selected_tool="embed_text",
            planned_tools=["embed_text"],
            steps=[
                AgentStep(name="inspect_input", status="completed", detail="Parsed request."),
                AgentStep(name="select_tool", status="completed", detail="Matched embed intent."),
                AgentStep(name="run_tool", status="completed", detail="Embedded text."),
            ],
            final_answer="已完成向量化。",
            session_id=None,
            memory_used=False,
            approval_required=False,
            approval_message=None,
            tool_input={"input_text": "请把这句话转成向量"},
            tool_output={"dimensions": 3},
        )

    monkeypatch.setattr(main_module, "run_agent", mock_run_agent)
    response = client.post("/agent", json={"input": "请把这句话转成向量"})

    assert response.status_code == 200
    body = response.json()
    assert body["selected_tool"] == "embed_text"
    assert body["tool_output"]["dimensions"] == 3


def test_agent_validation() -> None:
    response = client.post("/agent", json={"input": ""})
    assert response.status_code == 422


def test_agent_endpoint_returns_pending_confirmation(monkeypatch) -> None:
    async def mock_run_agent(_payload) -> AgentResponse:
        return AgentResponse(
            status="needs_confirmation",
            run_id="run-risk",
            input="请清空这个会话的记忆",
            selected_tool="clear_session_memory",
            planned_tools=["clear_session_memory"],
            steps=[
                AgentStep(name="inspect_input", status="completed", detail="Parsed request."),
                AgentStep(name="await_confirmation", status="pending", detail="Need confirmation."),
            ],
            final_answer="该操作需要确认。",
            session_id="demo-risk",
            memory_used=False,
            approval_required=True,
            approval_message="该操作需要确认。",
            tool_input={"session_id": "demo-risk"},
            tool_output=None,
        )

    monkeypatch.setattr(main_module, "run_agent", mock_run_agent)
    response = client.post("/agent", json={"input": "请清空这个会话的记忆", "session_id": "demo-risk"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_confirmation"
    assert body["approval_required"] is True


def test_agent_runs_endpoint_returns_recorded_runs() -> None:
    response = client.post("/agent", json={"input": "请计算 23 * 7"})
    runs_response = client.get("/agent/runs")

    assert response.status_code == 200
    assert runs_response.status_code == 200
    body = runs_response.json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["selected_tool"] == "calculator"
    assert body["runs"][0]["status"] == "completed"
