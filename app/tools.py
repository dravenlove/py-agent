from __future__ import annotations

import ast
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.embedding_client import generate_embedding
from app.memory import memory_store
from app.rerank_client import generate_rerank

ToolRunner = Callable[..., Awaitable["ToolExecutionResult"]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    runner: ToolRunner


@dataclass(frozen=True)
class ToolExecutionResult:
    summary: str
    output: dict[str, Any]


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


async def run_embed_text(input_text: str) -> ToolExecutionResult:
    model_name, vector = await generate_embedding(input_text)
    return ToolExecutionResult(
        summary=f"Embedded input text with {model_name} into a {len(vector)}-dimensional vector.",
        output={
            "model": model_name,
            "dimensions": len(vector),
            "embedding_head": vector[:5],
        },
    )


async def run_rerank_documents(query: str, documents: list[str], top_n: int | None = None) -> ToolExecutionResult:
    model_name, results = await generate_rerank(query, documents, top_n)
    return ToolExecutionResult(
        summary=f"Reranked {len(documents)} documents with {model_name}.",
        output={
            "model": model_name,
            "results": results,
            "best_match": results[0] if results else None,
        },
    )


async def run_calculator(expression: str) -> ToolExecutionResult:
    result = _evaluate_expression(expression)
    return ToolExecutionResult(
        summary=f"Calculated {expression} = {result}.",
        output={"expression": expression, "result": result},
    )


async def run_summarize_text(text: str) -> ToolExecutionResult:
    summary = _summarize_text(text)
    return ToolExecutionResult(
        summary="Summarized the selected text snippet.",
        output={"source_text": text, "summary": summary},
    )


async def run_clear_session_memory(session_id: str) -> ToolExecutionResult:
    deleted_count = memory_store.clear_session(session_id)
    return ToolExecutionResult(
        summary=f"Cleared {deleted_count} stored interaction(s) from session memory.",
        output={"session_id": session_id, "deleted_count": deleted_count},
    )


def get_tool_registry() -> dict[str, ToolDefinition]:
    return {
        "embed_text": ToolDefinition(
            name="embed_text",
            description="Turn input text into an embedding vector for retrieval workflows.",
            runner=run_embed_text,
        ),
        "rerank_documents": ToolDefinition(
            name="rerank_documents",
            description="Sort candidate documents by relevance to a query.",
            runner=run_rerank_documents,
        ),
        "calculator": ToolDefinition(
            name="calculator",
            description="Safely evaluate a basic arithmetic expression.",
            runner=run_calculator,
        ),
        "summarize_text": ToolDefinition(
            name="summarize_text",
            description="Create a short summary from a selected text snippet.",
            runner=run_summarize_text,
        ),
        "clear_session_memory": ToolDefinition(
            name="clear_session_memory",
            description="Delete all remembered interactions for a session after explicit confirmation.",
            runner=run_clear_session_memory,
        ),
    }


def list_tools() -> list[dict[str, str]]:
    return [
        {"name": tool.name, "description": tool.description}
        for tool in get_tool_registry().values()
    ]


def _evaluate_expression(expression: str) -> float | int:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid calculator expression.") from exc

    result = _evaluate_node(parsed.body)
    if result.is_integer():
        return int(result)
    return result


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate_node(node.operand)
        return _UNARY_OPERATORS[type(node.op)](operand)

    raise ValueError("Unsupported calculator expression.")


def _summarize_text(text: str, max_sentences: int = 2, max_chars: int = 180) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        raise ValueError("Cannot summarize empty text.")

    sentences = [
        part.strip()
        for part in normalized.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").splitlines()
        if part.strip()
    ]
    if not sentences:
        sentences = [normalized]

    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = f"{summary[: max_chars - 3].rstrip()}..."
    return summary
