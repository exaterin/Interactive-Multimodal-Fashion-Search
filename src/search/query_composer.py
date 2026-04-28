"""
Query Composer (LLM #1): turns (search_state + user message) into a single
CLIP-friendly retrieval query, plus reset / new_query flags.

Pipeline position:
  user_message → THIS LLM → query → retrieval_1 → catalog evidence
  → Response Generator (LLM #2) → updated_query → retrieval_2 → UI

Runs BEFORE retrieval, so it does NOT see catalog evidence. The grounded
refinement happens in the Response Generator (LLM #2), which emits an
updated_query that triggers a re-retrieval before results reach the UI.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from src.conversation.llm_client import LLMClient
from src.prompts import load_prompt
from src.search.search_state import SearchState
import src.log as log


SYSTEM_PROMPT = load_prompt("query_composer")


@dataclass
class ComposedQuery:
    query: str
    reset: bool = False
    new_query: bool = False


def compose_query(
    user_message: str,
    search_state: SearchState,
    llm_client: LLMClient,
    chat_history: Optional[List[dict]] = None,
) -> ComposedQuery:
    user_content = (
        f"Current search state:\n{search_state.to_context_str()}\n\n"
        f"User's new message: \"{user_message}\"\n\n"
        "Produce the retrieval query as JSON."
    )

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    log.query_compose_prompt(user_content)

    raw = ""
    try:
        raw = llm_client.chat(messages)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)
        result = ComposedQuery(
            query=str(data.get("query", "")).strip(),
            reset=bool(data.get("reset", False)),
            new_query=bool(data.get("new_query", False)),
        )
        log.query_compose_result(result)
        return result
    except (json.JSONDecodeError, Exception) as exc:
        fallback = ComposedQuery(
            query=search_state.current_query or user_message,
            reset=False,
            new_query=False,
        )
        log.query_compose_fallback(raw, str(exc), fallback)
        return fallback
