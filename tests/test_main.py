from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "day": 2}


def test_chat_with_mocked_llm(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "generate_reply", lambda _: "mocked answer")
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "mocked answer"
    assert "model" in body


def test_chat_validation() -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_embeddings_with_mocked_client(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "generate_embedding", lambda _: ("mock-emb-model", [0.1, 0.2, 0.3]))
    response = client.post("/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock-emb-model"
    assert body["dimensions"] == 3
    assert body["embedding"] == [0.1, 0.2, 0.3]


def test_embeddings_validation() -> None:
    response = client.post("/embeddings", json={"input": ""})
    assert response.status_code == 422
