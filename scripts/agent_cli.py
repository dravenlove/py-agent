import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


def main() -> None:
    default_url = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000/agent")
    parser = argparse.ArgumentParser(description="Call the local Day 4 agent endpoint.")
    parser.add_argument("--url", default=default_url, help="Agent endpoint URL.")
    parser.add_argument("--input", default=None, help="Agent task input. If omitted, prompt from stdin.")
    parser.add_argument("--doc", action="append", default=[], help="Candidate document text for rerank tasks.")
    parser.add_argument("--top-n", type=int, default=None, help="Optional top_n for rerank tasks.")
    parser.add_argument("--session-id", default=None, help="Optional session id for memory-aware agent runs.")
    args = parser.parse_args()

    user_input = args.input or input("Agent task: ").strip()
    if not user_input:
        raise SystemExit("Agent input cannot be empty.")

    payload: dict[str, object] = {"input": user_input}
    documents = [item.strip() for item in args.doc if item.strip()]
    if documents:
        payload["documents"] = documents
    if args.top_n is not None:
        payload["top_n"] = args.top_n
    if args.session_id:
        payload["session_id"] = args.session_id

    preview = {
        "method": "POST",
        "url": args.url,
        "headers": {"Content-Type": "application/json"},
        "json": payload,
    }
    print("Request preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    try:
        response = httpx.post(args.url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code} from {args.url}", file=sys.stderr)
        detail = exc.response.text.strip()
        if detail:
            print(f"Response: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
