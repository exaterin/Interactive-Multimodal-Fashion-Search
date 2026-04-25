from __future__ import annotations

import json
import re
from typing import List, Tuple, Union

from src.conversation.llm_client import LLMClient
from src.search.grounding import GroundingContext
from src.search.search_state import SearchState
from src.prompts import load_prompt
import src.log as log


_SYSTEM_PROMPT = load_prompt("grounded_response")


def generate_grounded_response(
    user_message: str,
    search_state: SearchState,
    grounding_context: GroundingContext,
    llm_client: LLMClient,
    chat_history: List[dict] = None,
) -> Tuple[str, List[str], str, dict]:
    """
    Generate a grounded assistant response using actual retrieval context.

    Relevance feedback, when present, is embedded in grounding_context.feedback_context
    and included automatically in the LLM prompt.

    Returns:
        (response_text, suggestions, updated_query, raw_llm_data)
    """
    state_section = f"Current search state:\n{search_state.to_context_str()}\n\n"
    instruction = (
        f"\n\nUser's message: \"{user_message}\"\n\n"
        "Respond using ONLY information present in the retrieved results above."
    )

    if grounding_context.is_multimodal:
        current_user_content: Union[str, list] = [
            {"type": "text", "text": state_section + "What was actually retrieved from the catalog:"},
            *grounding_context.to_multimodal_blocks(),
            {"type": "text", "text": instruction},
        ]
        log_user = "[multimodal content]"
    else:
        current_user_content = (
            state_section
            + "What was actually retrieved from the catalog:\n"
            + grounding_context.to_prompt_str()
            + instruction
        )
        log_user = current_user_content

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": current_user_content})

    log.llm_prompt(_SYSTEM_PROMPT, log_user)

    raw = ""
    try:
        raw = llm_client.chat(messages)
        log.llm_raw(raw)

        # Strip markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)

        log.llm_parsed(data)

        response = data.get("response", "").strip()
        suggestions = [s for s in data.get("suggestions", []) if s][:4]
        updated_query = data.get("updated_query", search_state.current_query).strip()

        return response, suggestions, updated_query, data

    except (json.JSONDecodeError, Exception) as exc:
        log.llm_fallback(raw, str(exc))
        fallback_text = raw if raw else "I found some results. Let me know how you'd like to refine."
        return fallback_text, [], search_state.current_query, {}
