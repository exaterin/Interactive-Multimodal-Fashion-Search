"""
Prompt Orchestration: bundles inputs into messages for the final responder LLM.

Inputs (matches the "Prompt Orchestration" box in the project schema):
  - Conversational Feedback (the user's current message + chat history)
  - Current Search State
  - Catalog Evidence (built from the FIRST retrieval — Composer's query)
  - Preference Evidence

The responder also emits updated_query, which triggers a re-retrieval before
the products are returned to the UI.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from src.prompts import load_prompt
from src.search.catalog_evidence import CatalogEvidence
from src.search.preference_evidence import PreferenceEvidence
from src.search.search_state import SearchState


SYSTEM_PROMPT = load_prompt("grounded_response")

_INSTRUCTION_TEMPLATE = (
    "\n\nUser's message: \"{message}\"\n\n"
    "Respond using ONLY information present in the catalog and preference evidence above."
)


def orchestrate_messages(
    user_message: str,
    search_state: SearchState,
    catalog_evidence: CatalogEvidence,
    preference_evidence: PreferenceEvidence,
    chat_history: List[dict] = None,
) -> Tuple[List[dict], Union[str, list]]:
    """
    Returns:
        (messages, log_user_content) — log_user_content is a debug-friendly
        string view of the current-turn user content.
    """
    state_section = f"Current search state:\n{search_state.to_context_str()}\n\n"
    instruction = _INSTRUCTION_TEMPLATE.format(message=user_message)
    is_multimodal = catalog_evidence.is_multimodal or preference_evidence.is_multimodal

    if is_multimodal:
        current_user_content: Union[str, list] = [
            {"type": "text", "text": state_section},
            *catalog_evidence.to_blocks(),
            *preference_evidence.to_blocks(),
            {"type": "text", "text": instruction},
        ]
        log_user = "[multimodal content]"
    else:
        parts = [state_section, catalog_evidence.to_text()]
        if not preference_evidence.is_empty:
            parts.append("")
            parts.append(preference_evidence.to_text())
        parts.append(instruction)
        current_user_content = "\n".join(parts)
        log_user = current_user_content

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": current_user_content})

    return messages, log_user
