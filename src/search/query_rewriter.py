"""
Query Rewriter (LLM #1): turn (search_state + user message) into a single
CLIP-friendly retrieval query, plus a reset flag.

Runs BEFORE retrieval, so the catalog evidence is always built from the
right query the first time. No suggestions, no constraints, no intent
classification — that is the final responder LLM's job.
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


SYSTEM_PROMPT = load_prompt("query_rewriter")


@dataclass
class RewrittenQuery:
    query: str
    reset: bool = False
    topic_switch: bool = False


def rewrite_query(
    user_message: str,
    search_state: SearchState,
    llm_client: LLMClient,
    chat_history: Optional[List[dict]] = None,
) -> RewrittenQuery:
    """
    Returns a RewrittenQuery. On parse failure, falls back to:
      - reset=False, query=current_query if non-empty, else user_message.
    """
    state_str = search_state.to_context_str()
    user_content = (
        f"Current search state:\n{state_str}\n\n"
        f"User's new message: \"{user_message}\"\n\n"
        "Produce the rewritten retrieval query as JSON."
    )

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    log.query_rewrite_prompt(user_content)

    raw = ""
    try:
        raw = llm_client.chat(messages)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)
        result = RewrittenQuery(
            query=str(data.get("query", "")).strip(),
            reset=bool(data.get("reset", False)),
            topic_switch=bool(data.get("topic_switch", False)),
        )
        log.query_rewrite_result(result, raw)
        return result
    except (json.JSONDecodeError, Exception) as exc:
        fallback = RewrittenQuery(
            query=search_state.current_query or user_message,
            reset=False,
        )
        log.query_rewrite_fallback(raw, str(exc), fallback)
        return fallback
