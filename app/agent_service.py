from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.memory import memory_store
from app.schemas import AgentRequest, AgentResponse, AgentStep
from app.tools import get_tool_registry, list_tools

logger = logging.getLogger("ai_agent.agent")

_EMBED_KEYWORDS = ("向量", "embedding", "embed", "嵌入")
_SUMMARY_KEYWORDS = ("总结", "概括", "摘要", "summary", "summarize")
_RERANK_HINTS = ("重排", "排序", "相关", "最相关", "比较", "compare")
_MEMORY_DOCUMENT_HINTS = ("这些文档", "刚才的文档", "上面的文档", "继续比较")
_EXPRESSION_PATTERN = re.compile(r"[0-9\.\s\+\-\*\/%\(\)]+")
_CALCULATOR_KEYWORDS = ("计算", "算一下", "等于", "多少", "calc", "calculator")


@dataclass(frozen=True)
class PlannedAction:
    tool_name: str
    tool_input: dict[str, Any]
    detail: str


async def run_agent(payload: AgentRequest) -> AgentResponse:
    steps = [AgentStep(name="inspect_input", status="completed", detail="Parsed the incoming agent request.")]
    memory_used = False
    documents = payload.documents

    if payload.session_id:
        recent_memory = memory_store.get_recent(payload.session_id)
        if recent_memory:
            steps.append(
                AgentStep(
                    name="load_memory",
                    status="completed",
                    detail=f"Loaded {len(recent_memory)} recent interaction(s) from session memory.",
                )
            )
        remembered_documents = memory_store.get_last_documents(payload.session_id)
        if not documents and remembered_documents and _mentions_memory_documents(payload.input):
            documents = remembered_documents
            memory_used = True
            steps.append(
                AgentStep(
                    name="reuse_memory",
                    status="completed",
                    detail=f"Reused {len(remembered_documents)} remembered document(s) from session memory.",
                )
            )

    planning_payload = payload.model_copy(update={"documents": documents})
    planned_actions = _plan_actions(planning_payload)
    if not planned_actions:
        steps.append(
            AgentStep(
                name="finish",
                status="completed",
                detail="Returned supported capabilities because no tool matched the request.",
            )
        )
        response = AgentResponse(
            input=payload.input,
            selected_tool="unsupported",
            planned_tools=[],
            steps=steps,
            final_answer=_build_unsupported_message(),
            session_id=payload.session_id,
            memory_used=memory_used,
            tool_input=None,
            tool_output={"available_tools": list_tools()},
        )
        _remember(payload, response)
        return response

    steps.append(AgentStep(name="select_tool", status="completed", detail=planned_actions[0].detail))
    steps.append(
        AgentStep(
            name="plan",
            status="completed",
            detail=f"Planned {len(planned_actions)} action(s): {', '.join(action.tool_name for action in planned_actions)}.",
        )
    )

    registry = get_tool_registry()
    first_tool_input: dict[str, Any] | None = None
    current_output: dict[str, Any] | None = None

    for action in planned_actions:
        logger.info("Selected tool %s for input: %s", action.tool_name, payload.input)
        tool = registry[action.tool_name]
        tool_input = _resolve_tool_input(action.tool_input, current_output)
        if first_tool_input is None:
            first_tool_input = dict(tool_input)
        result = await tool.runner(**tool_input)
        current_output = result.output
        steps.append(AgentStep(name=f"run_{action.tool_name}", status="completed", detail=result.summary))

    assert current_output is not None
    selected_tool = planned_actions[0].tool_name
    response = AgentResponse(
        input=payload.input,
        selected_tool=selected_tool,
        planned_tools=[action.tool_name for action in planned_actions],
        steps=steps,
        final_answer=_build_final_answer(planned_actions[-1].tool_name, current_output),
        session_id=payload.session_id,
        memory_used=memory_used,
        tool_input=first_tool_input,
        tool_output=current_output,
    )
    _remember(payload, response)
    return response


def _plan_actions(payload: AgentRequest) -> list[PlannedAction]:
    text = payload.input.strip()
    lowered = text.lower()
    expression = _extract_expression(text)

    if expression and (_contains_any(lowered, _CALCULATOR_KEYWORDS) or _looks_like_expression(text, expression)):
        return [
            PlannedAction(
                tool_name="calculator",
                tool_input={"expression": expression},
                detail=f"Matched calculator intent from expression `{expression}`.",
            )
        ]

    if payload.documents:
        actions = [
            PlannedAction(
                tool_name="rerank_documents",
                tool_input={"query": text, "documents": payload.documents, "top_n": payload.top_n},
                detail=f"Detected {len(payload.documents)} candidate documents and selected rerank_documents.",
            )
        ]
        if _contains_any(lowered, _SUMMARY_KEYWORDS):
            actions.append(
                PlannedAction(
                    tool_name="summarize_text",
                    tool_input={"text": {"from_output": "best_match.document"}},
                    detail="Added summarize_text because the request asked for a summary.",
                )
            )
        return actions

    if _contains_any(lowered, _EMBED_KEYWORDS):
        return [
            PlannedAction(
                tool_name="embed_text",
                tool_input={"input_text": text},
                detail="Matched embedding intent from vector-related keywords.",
            )
        ]

    return []


def _resolve_tool_input(tool_input: dict[str, Any], current_output: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in tool_input.items():
        if isinstance(value, dict) and "from_output" in value:
            if current_output is None:
                raise ValueError("Tool plan expected previous output, but none was available.")
            resolved[key] = _read_path(current_output, str(value["from_output"]))
        else:
            resolved[key] = value
    return resolved


def _read_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, dict):
            raise ValueError(f"Cannot resolve tool output path `{path}`.")
        current = current.get(segment)
    return current


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _mentions_memory_documents(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, tuple(item.lower() for item in _MEMORY_DOCUMENT_HINTS)) or _contains_any(text, _RERANK_HINTS)


def _extract_expression(text: str) -> str | None:
    candidates = [chunk.strip() for chunk in _EXPRESSION_PATTERN.findall(text)]
    valid = [chunk for chunk in candidates if re.search(r"\d", chunk) and re.search(r"[\+\-\*\/%]", chunk)]
    if not valid:
        return None
    return max(valid, key=len)


def _looks_like_expression(text: str, expression: str) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    compact_expression = re.sub(r"\s+", "", expression)
    return compact_text == compact_expression


def _build_final_answer(last_tool_name: str, tool_output: dict[str, Any]) -> str:
    if last_tool_name == "embed_text":
        return (
            f"已完成向量化，使用模型 {tool_output['model']}，"
            f"向量维度 {tool_output['dimensions']}，前 5 个值为 {tool_output['embedding_head']}。"
        )

    if last_tool_name == "rerank_documents":
        best_match = tool_output.get("best_match")
        if isinstance(best_match, dict) and best_match:
            return (
                f"已完成重排，最相关的是第 {best_match.get('index')} 段文档，"
                f"分数 {best_match.get('score')}，内容是：{best_match.get('document')}"
            )
        return "已完成重排，但没有返回有效结果。"

    if last_tool_name == "summarize_text":
        return f"我先完成了文档筛选，再总结结果：{tool_output['summary']}"

    if last_tool_name == "calculator":
        return f"计算完成：{tool_output['expression']} = {tool_output['result']}"

    return "工具执行完成。"


def _build_unsupported_message() -> str:
    available = ", ".join(tool["name"] for tool in list_tools())
    return f"当前这个 Day 5 Agent 只支持这些工具：{available}。你可以让我做向量化、文档重排、文档总结，或计算数学表达式。"


def _remember(payload: AgentRequest, response: AgentResponse) -> None:
    if not payload.session_id:
        return
    memory_store.append_interaction(
        session_id=payload.session_id,
        user_input=payload.input,
        planned_tools=response.planned_tools,
        tool_input=response.tool_input,
        tool_output=response.tool_output,
        final_answer=response.final_answer,
    )
