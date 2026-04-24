from __future__ import annotations

import json
import re
from typing import List, Tuple

from src.conversation.llm_client import LLMClient
from src.search.grounding_analyzer import GroundingContext
from src.search.search_state import SearchState
from src.prompts import load_prompt
import src.log as log


_SYSTEM_PROMPT = load_prompt("grounded_response")


def generate_grounded_response(
    user_message: str,
    search_state: SearchState,
    grounding_context: GroundingContext,
    llm_client: LLMClient,
    liked_context: str = "",
) -> Tuple[str, List[str], str, dict]:
    """
    Generate a grounded assistant response using actual retrieval context.

    Returns:
        (response_text, suggestions, updated_query, raw_llm_data)
    """
    liked_section = f"\n\n{liked_context}" if liked_context else ""
    user_content = (
        f"Current search state:\n{search_state.to_context_str()}\n\n"
        f"What was actually retrieved from the catalog:\n"
        f"{grounding_context.to_prompt_str()}"
        f"{liked_section}\n\n"
        f"User's message: \"{user_message}\"\n\n"
        f"Respond using ONLY information present in the retrieved results above."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    log.llm_prompt(_SYSTEM_PROMPT, user_content)

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
