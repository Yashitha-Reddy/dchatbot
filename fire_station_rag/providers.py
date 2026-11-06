"""Provider configuration shared by the CLI and HTTP service."""
import os
import json
import logging
import re

LOG = logging.getLogger(__name__)


def create_llm():
    from settings import load_settings
    load_settings()
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("Set NVIDIA_API_KEY before using the NVIDIA provider")
    model = os.getenv("LLM_MODEL") or "nvidia/nemotron-3.5-lightning-30b-a3b"
    return ChatNVIDIA(model=model, temperature=0, max_tokens=2048,
                      timeout=120,
                      model_kwargs={"chat_template_kwargs": {"enable_thinking": False}})


def structured_query(llm, schema, messages):
    # Use prompt-based JSON on NVIDIA: this endpoint rejects legacy guided_json.
    # Local schema validation remains mandatory before any data operation.
    # Strip legacy prompt control tokens before sending messages.
    cleaned = [m.model_copy(update={"content": m.content.replace("/no_think", "")})
               if isinstance(m.content, str) else m for m in messages]
    from langchain_core.messages import SystemMessage
    instruction = SystemMessage(content=(
        "Return exactly one JSON object matching this schema. No explanation, "
        "reasoning, markdown or extra keys. Use null for unused optional values.\n"
        + json.dumps(schema.model_json_schema())
    ))
    from langchain_core.messages import HumanMessage
    request = [instruction, *cleaned]
    for attempt in range(2):
        response = llm.invoke(request)
        content = response.content.strip()
        # Some reasoning models return a closed thinking block before final JSON.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return schema.model_validate_json(content)
        except ValueError as exc:
            errors = exc.errors(include_input=False, include_url=False) if hasattr(exc, "errors") else []
            # Log validation locations, not raw model reasoning, history, or secrets.
            issues = [{"type": e["type"], "loc": e["loc"]} for e in errors]
            LOG.warning("Invalid %s output attempt=%s chars=%s finish_reason=%s issues=%s",
                        schema.__name__, attempt + 1, len(content),
                        getattr(response, "response_metadata", {}).get("finish_reason"), issues)
            if attempt:
                raise
            request = [*request, HumanMessage(content=(
                "Your previous response failed validation. Return a fresh JSON object only. "
                "Use the exact schema field names and enum values; no thinking or prose. "
                "Validation issues: " + json.dumps(issues)
            ))]
