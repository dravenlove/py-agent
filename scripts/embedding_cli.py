import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


def _mask_token(token: str) -> str:
    if len(token) <= 10:
        return "****"
    return f"{token[:6]}...{token[-4:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Call embeddings endpoint with dedicated env config.")
    parser.add_argument(
        "--text",
        default=None,
        help="Input text for embedding. If omitted, prompt from stdin.",
    )
    args = parser.parse_args()

    text = args.text or input("Text: ").strip()
    if not text:
        raise SystemExit("Text cannot be empty.")

    api_key = os.getenv("EMBEDDING_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("EMBEDDING_OPENAI_API_KEY is missing. Please set it in .env.")

    base_url = os.getenv("EMBEDDING_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://router.tumuer.me/v1"
    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    encoding_format = os.getenv("EMBEDDING_ENCODING_FORMAT", "float")

    payload = {
        "model": model,
        "input": text,
        "encoding_format": encoding_format,
    }

    preview = {
        "method": "POST",
        "url": f"{base_url.rstrip('/')}/embeddings",
        "headers": {
            "Authorization": f"Bearer {_mask_token(api_key)}",
            "Content-Type": "application/json",
        },
        "json": payload,
    }
    print("Request preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(**payload)
    vector = response.data[0].embedding

    output = {
        "model": response.model,
        "dimensions": len(vector),
        "embedding_head": vector[:10],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

