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
    default_url = os.getenv("AGENT_RUNS_API_URL", "http://127.0.0.1:8000/agent/runs")
    parser = argparse.ArgumentParser(description="Inspect Day 7 agent run history.")
    parser.add_argument("--url", default=default_url, help="Agent runs endpoint URL.")
    parser.add_argument("--session-id", default=None, help="Optional session id filter.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to fetch.")
    args = parser.parse_args()

    params: dict[str, object] = {"limit": args.limit}
    if args.session_id:
        params["session_id"] = args.session_id

    preview = {
        "method": "GET",
        "url": args.url,
        "params": params,
    }
    print("Request preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    try:
        response = httpx.get(args.url, params=params, timeout=30)
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
