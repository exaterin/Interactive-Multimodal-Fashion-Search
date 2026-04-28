"""
Response Generator (LLM #2): the conversational, grounded responder.

Pipeline position:
  retrieval_1 (Composer's query) → catalog evidence → THIS LLM
  → updated_query → retrieval_2 (if changed) → UI

Sees catalog evidence (from the first retrieval) and emits the conversational
reply, grounded suggestions, intent, full constraint set, AND an updated_query
that drives a re-retrieval before results reach the UI.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

from src.conversation.llm_client import LLMClient
from src.search.catalog_evidence import CatalogEvidence
from src.search.preference_evidence import PreferenceEvidence
from src.search.prompt_orchestration import SYSTEM_PROMPT, orchestrate_messages
from src.search.search_state import SearchState
import src.log as log


def generate_grounded_response(
    user_message: str,
    search_state: SearchState,
    catalog_evidence: CatalogEvidence,
    preference_evidence: PreferenceEvidence,
    llm_client: LLMClient,
    chat_history: List[dict] = None,
) -> Tuple[str, List[str], str, dict]:
    """
    Returns:
        (response_text, suggestions, updated_query, raw_llm_data)
    """
    messages, log_user = orchestrate_messages(
        user_message=user_message,
        search_state=search_state,
        catalog_evidence=catalog_evidence,
        preference_evidence=preference_evidence,
        chat_history=chat_history,
    )

    log.llm_prompt(SYSTEM_PROMPT, log_user)

    raw = ""
    try:
        raw = llm_client.chat(messages)
        log.llm_raw(raw)

        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)

        log.llm_parsed(data)

        response = data.get("response", "").strip()
        suggestions = [s for s in data.get("suggestions", []) if s][:4]
        updated_query = str(data.get("updated_query", "")).strip()
        return response, suggestions, updated_query, data

    except (json.JSONDecodeError, Exception) as exc:
        log.llm_fallback(raw, str(exc))
        fallback_text = raw if raw else "I found some results. Let me know how you'd like to refine."
        return fallback_text, [], search_state.current_query, {}
