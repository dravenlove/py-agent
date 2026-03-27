from __future__ import annotations

from threading import Lock
from typing import Any


class AgentRunStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: list[dict[str, Any]] = []

    def append_run(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._runs.append(dict(record))

    def list_runs(self, *, limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            runs = self._runs
            if session_id:
                runs = [item for item in runs if item.get("session_id") == session_id]
            selected = runs[-limit:]
            return [dict(item) for item in reversed(selected)]

    def reset(self) -> None:
        with self._lock:
            self._runs = []


agent_run_store = AgentRunStore()
