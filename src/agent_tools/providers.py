from .types import ToolCall, ToolResult


def _as_dict(obj):
    """Accept plain dicts or SDK response objects (OpenAI/Anthropic SDKs use Pydantic)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"cannot read response of type {type(obj).__name__}")


def _openai_message(response):
    data = _as_dict(response)
    return data["choices"][0]["message"] if "choices" in data else data


def parse_tool_calls(response, provider="openai"):
    if provider == "anthropic":
        blocks = _as_dict(response).get("content", [])
        return [ToolCall(b["id"], b["name"], b.get("input") or {})
                for b in blocks if b.get("type") == "tool_use"]
    msg = _openai_message(response)
    return [ToolCall(tc["id"], tc["function"]["name"], tc["function"].get("arguments") or "{}")
            for tc in msg.get("tool_calls") or []]


def assistant_message(response, provider="openai"):
    """The assistant turn to append to `messages` BEFORE the tool results."""
    if provider == "anthropic":
        return {"role": "assistant", "content": _as_dict(response)["content"]}
    msg = _openai_message(response)
    out = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def results_to_messages(results, provider="openai"):
    if not results:
        return []
    if provider == "anthropic":
        # Anthropic wants ALL results for one turn inside ONE user message
        return [{"role": "user", "content": [r.to_message("anthropic") for r in results]}]
    return [r.to_message("openai") for r in results]


def get_text(response, provider="openai"):
    if provider == "anthropic":
        blocks = _as_dict(response).get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return _openai_message(response).get("content") or ""