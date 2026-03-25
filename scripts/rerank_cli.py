import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


def _mask_token(token: str) -> str:
    if len(token) <= 10:
        return "****"
    return f"{token[:6]}...{token[-4:]}"


def _extract_doc_text(doc: object, fallback: str) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        text = doc.get("text")
        if isinstance(text, str):
            return text
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Call rerank endpoint with dedicated env config.")
    parser.add_argument("--query", default=None, help="Query text. If omitted, prompt from stdin.")
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="Candidate document text. Repeat this flag for multiple docs.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Optional top_n for rerank.")
    args = parser.parse_args()

    query = args.query or input("Query: ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    documents = [d.strip() for d in args.doc if d.strip()]
    if not documents:
        raise SystemExit("At least one --doc is required.")

    api_key = os.getenv("RERANK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("RERANK_OPENAI_API_KEY is missing. Please set it in .env.")

    base_url = os.getenv("RERANK_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://router.tumuer.me/v1"
    model = os.getenv("RERANK_MODEL", "jina-reranker-v3")

    payload: dict[str, object] = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if args.top_n is not None:
        payload["top_n"] = args.top_n

    url = f"{base_url.rstrip('/')}/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    preview = {
        "method": "POST",
        "url": url,
        "headers": {
            "Authorization": f"Bearer {_mask_token(api_key)}",
            "Content-Type": "application/json",
        },
        "json": payload,
    }
    print("Request preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code} from {url}", file=sys.stderr)
        print(exc.response.text, file=sys.stderr)
        raise SystemExit(1) from exc

    body = response.json()
    results = body.get("results", []) if isinstance(body, dict) else []
    normalized: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("relevance_score")
        doc_obj = item.get("document")
        fallback = ""
        if isinstance(idx, int) and 0 <= idx < len(documents):
            fallback = documents[idx]
        text = _extract_doc_text(doc_obj, fallback)
        normalized.append(
            {
                "index": idx,
                "score": score,
                "document": text,
            }
        )

    output = {
        "model": model,
        "query": query,
        "results": normalized,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

