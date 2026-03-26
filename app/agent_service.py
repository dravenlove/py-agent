from __future__ import annotations

import logging
import re

from app.schemas import AgentRequest, AgentResponse, AgentStep
from app.tools import ToolExecutionResult, get_tool_registry, list_tools

logger = logging.getLogger("ai_agent.agent")

_EMBED_KEYWORDS = ("向量", "embedding", "embed", "嵌入")
_EXPRESSION_PATTERN = re.compile(r"[0-9\.\s\+\-\*\/%\(\)]+")
_CALCULATOR_KEYWORDS = ("计算", "算一下", "等于", "多少", "calc", "calculator")


async def run_agent(payload: AgentRequest) -> AgentResponse:
    steps = [
        AgentStep(name="inspect_input", status="completed", detail="Parsed the incoming agent request."),
    ]

    selected_tool, tool_input, reason = _select_tool(payload)
    steps.append(AgentStep(name="select_tool", status="completed", detail=reason))

    if not selected_tool:
        steps.append(
            AgentStep(
                name="finish",
                status="completed",
                detail="Returned supported capabilities because no tool matched the request.",
            )
        )
        return AgentResponse(
            input=payload.input,
            selected_tool="unsupported",
            steps=steps,
            final_answer=_build_unsupported_message(),
            tool_input=None,
            tool_output={"available_tools": list_tools()},
        )

    registry = get_tool_registry()
    tool = registry[selected_tool]
    logger.info("Selected tool %s for input: %s", selected_tool, payload.input)
    result = await tool.runner(**tool_input)
    steps.append(AgentStep(name="run_tool", status="completed", detail=result.summary))
    steps.append(AgentStep(name="finalize", status="completed", detail="Built the final answer from tool output."))

    return AgentResponse(
        input=payload.input,
        selected_tool=selected_tool,
        steps=steps,
        final_answer=_build_final_answer(selected_tool, result),
        tool_input=tool_input,
        tool_output=result.output,
    )


def _select_tool(payload: AgentRequest) -> tuple[str | None, dict[str, object], str]:
    text = payload.input.strip()
    lowered = text.lower()
    expression = _extract_expression(text)

    if expression and (_contains_any(lowered, _CALCULATOR_KEYWORDS) or _looks_like_expression(text, expression)):
        return "calculator", {"expression": expression}, f"Matched calculator intent from expression `{expression}`."

    if payload.documents:
        return (
            "rerank_documents",
            {"query": text, "documents": payload.documents, "top_n": payload.top_n},
            f"Detected {len(payload.documents)} candidate documents and selected rerank_documents.",
        )

    if _contains_any(lowered, _EMBED_KEYWORDS):
        return "embed_text", {"input_text": text}, "Matched embedding intent from vector-related keywords."

    return None, {}, "No matching tool intent found."


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_expression(text: str) -> str | None:
    candidates = [chunk.strip() for chunk in _EXPRESSION_PATTERN.findall(text)]
    valid = [
        chunk
        for chunk in candidates
        if re.search(r"\d", chunk) and re.search(r"[\+\-\*\/%]", chunk)
    ]
    if not valid:
        return None
    return max(valid, key=len)


def _looks_like_expression(text: str, expression: str) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    compact_expression = re.sub(r"\s+", "", expression)
    return compact_text == compact_expression


def _build_final_answer(selected_tool: str, result: ToolExecutionResult) -> str:
    if selected_tool == "embed_text":
        return (
            f"已完成向量化，使用模型 {result.output['model']}，"
            f"向量维度 {result.output['dimensions']}，前 5 个值为 {result.output['embedding_head']}。"
        )

    if selected_tool == "rerank_documents":
        best_match = result.output.get("best_match")
        if isinstance(best_match, dict) and best_match:
            return (
                f"已完成重排，最相关的是第 {best_match.get('index')} 段文档，"
                f"分数 {best_match.get('score')}，内容是：{best_match.get('document')}"
            )
        return "已完成重排，但没有返回有效结果。"

    if selected_tool == "calculator":
        return f"计算完成：{result.output['expression']} = {result.output['result']}"

    return "工具执行完成。"


def _build_unsupported_message() -> str:
    available = ", ".join(tool["name"] for tool in list_tools())
    return f"当前这个 Day 4 Agent 只支持这些工具：{available}。你可以让我做向量化、文档重排，或计算数学表达式。"
