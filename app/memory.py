from __future__ import annotations

from threading import Lock
from typing import Any


class SessionMemoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    def append_interaction(
        self,
        *,
        session_id: str,
        user_input: str,
        planned_tools: list[str],
        tool_input: dict[str, Any] | None,
        tool_output: dict[str, Any] | None,
        final_answer: str,
    ) -> None:
        record = {
            "input": user_input,
            "planned_tools": planned_tools,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "final_answer": final_answer,
        }
        with self._lock:
            self._sessions.setdefault(session_id, []).append(record)

    def get_recent(self, session_id: str, limit: int = 3) -> list[dict[str, Any]]:
        with self._lock:
            history = self._sessions.get(session_id, [])
            return [dict(item) for item in history[-limit:]]

    def get_last_documents(self, session_id: str) -> list[str] | None:
        with self._lock:
            history = self._sessions.get(session_id, [])
            for item in reversed(history):
                tool_input = item.get("tool_input")
                if isinstance(tool_input, dict):
                    documents = tool_input.get("documents")
                    if isinstance(documents, list) and documents:
                        return [str(doc) for doc in documents]
        return None

    def reset(self) -> None:
        with self._lock:
            self._sessions = {}


memory_store = SessionMemoryStore()
