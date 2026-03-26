import argparse
import asyncio
import json
import statistics
import time

import httpx


def build_payload(endpoint: str, text: str) -> dict[str, object]:
    if endpoint == "chat":
        return {"message": text}
    if endpoint == "embeddings":
        return {"input": text}
    return {
        "query": text,
        "documents": [
            f"{text} - 候选文档 1",
            f"{text} - 候选文档 2",
            f"{text} - 候选文档 3",
        ],
        "top_n": 2,
    }


async def send_one(client: httpx.AsyncClient, url: str, payload: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": response.is_success,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except httpx.HTTPError:
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": latency_ms,
        }


async def worker(client: httpx.AsyncClient, url: str, payload: dict[str, object], semaphore: asyncio.Semaphore) -> dict[str, object]:
    async with semaphore:
        return await send_one(client, url, payload)


async def run_load_test(base_url: str, endpoint: str, text: str, count: int, concurrency: int) -> dict[str, object]:
    url = f"{base_url.rstrip('/')}/{endpoint}"
    payload = build_payload(endpoint, text)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60) as client:
        started = time.perf_counter()
        tasks = [worker(client, url, payload, semaphore) for _ in range(count)]
        results = await asyncio.gather(*tasks)
        total_ms = (time.perf_counter() - started) * 1000

    latencies = [item["latency_ms"] for item in results]
    successes = sum(1 for item in results if item["ok"])
    failures = len(results) - successes

    return {
        "url": url,
        "count": count,
        "concurrency": concurrency,
        "successes": successes,
        "failures": failures,
        "total_ms": round(total_ms, 2),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0,
        "payload_preview": payload,
        "statuses": [item["status_code"] for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple async load test against local FastAPI endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the local app.")
    parser.add_argument(
        "--endpoint",
        choices=["chat", "embeddings", "rerank"],
        default="embeddings",
        help="Endpoint to test.",
    )
    parser.add_argument("--text", default="Day 3 load test sample", help="Input text used to build the request body.")
    parser.add_argument("--count", type=int, default=10, help="Total number of requests.")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum number of in-flight requests.")
    args = parser.parse_args()

    report = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            endpoint=args.endpoint,
            text=args.text,
            count=args.count,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
