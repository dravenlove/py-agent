from fastapi.testclient import TestClient

import app.main as main_module
from app.errors import UpstreamAuthError, UpstreamNotFoundError, UpstreamServiceError, UpstreamTimeoutError
from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "day": 3}


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
