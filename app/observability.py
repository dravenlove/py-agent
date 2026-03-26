from __future__ import annotations

from contextvars import ContextVar, Token
from threading import Lock

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._requests_total = 0
            self._failures_total = 0
            self._routes: dict[str, dict[str, float | int]] = {}

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._requests_total += 1
            if status_code >= 400:
                self._failures_total += 1

            route = self._routes.setdefault(
                path,
                {
                    "requests": 0,
                    "failures": 0,
                    "total_latency_ms": 0.0,
                    "avg_latency_ms": 0.0,
                    "last_status_code": 0,
                },
            )
            route["requests"] += 1
            if status_code >= 400:
                route["failures"] += 1
            route["total_latency_ms"] += latency_ms
            route["avg_latency_ms"] = route["total_latency_ms"] / route["requests"]
            route["last_status_code"] = status_code

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            routes: dict[str, dict[str, float | int]] = {}
            for path, stats in self._routes.items():
                routes[path] = {
                    "requests": int(stats["requests"]),
                    "failures": int(stats["failures"]),
                    "avg_latency_ms": round(float(stats["avg_latency_ms"]), 2),
                    "last_status_code": int(stats["last_status_code"]),
                }
            return {
                "requests_total": self._requests_total,
                "failures_total": self._failures_total,
                "routes": routes,
            }


metrics_store = MetricsStore()


def set_request_id(request_id: str) -> Token[str]:
    return request_id_var.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    request_id_var.reset(token)


def get_request_id() -> str:
    return request_id_var.get()
