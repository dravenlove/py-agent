from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


@dataclass(frozen=True)
class Settings:
    chat_openai_api_key: str | None
    chat_openai_base_url: str | None
    chat_model: str
    chat_api_style: str
    embedding_openai_api_key: str | None
    embedding_openai_base_url: str | None
    embedding_model: str
    embedding_encoding_format: str


settings = Settings(
    chat_openai_api_key=os.getenv("CHAT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    chat_openai_base_url=os.getenv("CHAT_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    chat_model=os.getenv("CHAT_MODEL") or os.getenv("MODEL", "gpt-5.4-mini"),
    chat_api_style=os.getenv("CHAT_API_STYLE", "responses"),
    embedding_openai_api_key=os.getenv("EMBEDDING_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    embedding_openai_base_url=os.getenv("EMBEDDING_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"),
    embedding_encoding_format=os.getenv("EMBEDDING_ENCODING_FORMAT", "float"),
)
