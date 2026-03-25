import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


def _mask_headers(headers: dict[str, str]) -> dict[str, str]:
    masked = dict(headers)
    auth = masked.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.removeprefix("Bearer ").strip()
        if token:
            if len(token) <= 10:
                masked["Authorization"] = "Bearer ****"
            else:
                masked["Authorization"] = f"Bearer {token[:6]}...{token[-4:]}"
    return masked


def print_request_preview(url: str, headers: dict[str, str], payload: dict) -> None:
    preview = {
        "method": "POST",
        "url": url,
        "headers": _mask_headers(headers),
        "json": payload,
    }
    print("Request preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))


def build_headers() -> dict[str, str]:
    raw_content_type = os.getenv("CHAT_API_CONTENT_TYPE", "application/json").strip() or "application/json"
    content_type = raw_content_type
    if content_type.startswith("/") or "/" not in content_type:
        print(
            f"Invalid CHAT_API_CONTENT_TYPE={raw_content_type!r}, fallback to 'application/json'.",
            file=sys.stderr,
        )
        content_type = "application/json"

    headers = {"Content-Type": content_type}

    token = os.getenv("CHAT_API_BEARER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_payload(message: str) -> dict:
    return {"message": message}


def main() -> None:
    default_url = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat")
    parser = argparse.ArgumentParser(description="Call a chat or embeddings endpoint.")
    parser.add_argument(
        "--url",
        default=default_url,
        help="Target endpoint URL.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Message to send. If omitted, prompt from stdin.",
    )
    args = parser.parse_args()

    message = args.message or input("You: ").strip()
    if not message:
        raise SystemExit("Message cannot be empty.")
    if args.url.lower().endswith("/embeddings"):
        raise SystemExit("Use scripts/embedding_cli.py for embedding calls.")

    headers = build_headers()
    payload = build_payload(message)
    print_request_preview(args.url, headers, payload)

    try:
        response = httpx.post(args.url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        print(f"HTTP {status} from {args.url}", file=sys.stderr)
        if status == 404:
            print(
                "Tip: CHAT_API_URL should point to your chat endpoint (e.g. http://127.0.0.1:8000/chat), not just /v1.",
                file=sys.stderr,
            )
        if status == 401:
            print(
                "Tip: set CHAT_API_BEARER_TOKEN (or OPENAI_API_KEY) in .env so Authorization is sent.",
                file=sys.stderr,
            )
        detail = exc.response.text.strip()
        if detail:
            print(f"Response: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc

    payload = response.json()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
