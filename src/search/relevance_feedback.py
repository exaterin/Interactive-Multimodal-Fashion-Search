"""
Relevance Feedback pipeline.

Triggered when the user selects 1–3 items from the current retrieval and
optionally writes a short natural-language comment about them. Distinct from
the standard /chat pipeline:

  selected_items + comment + search_state
       → Relevance Feedback LLM
       → refined_query  + response  + updated state
       → Multimodal Retriever
       → UI

Selected items are treated as PARTIAL preference signals, not as a target to
clone. The user's comment, when provided, has priority over the raw item
attributes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from src.conversation.llm_client import LLMClient
from src.data.fashionpedia.catalog import FashionpediaCatalog
from src.prompts import load_prompt
from src.search.context_extraction import (
    ExtractionStrategy,
    ItemContext,
    REFINEMENT_SUPERCATS,
    get_formatter,
)
from src.search.preference_evidence import PreferenceItem
from src.search.search_state import SearchState
import src.log as log


SYSTEM_PROMPT = load_prompt("relevance_feedback")


@dataclass
class FeedbackResult:
    response: str
    refined_query: str
    raw: dict


def _build_selected_items_block(
    items: List[PreferenceItem],
    catalog: FashionpediaCatalog,
    strategy: ExtractionStrategy,
) -> Tuple[str, list, bool]:
    """
    Render the selected items the same way Catalog/Preference Evidence does,
    so the LLM sees them in a familiar shape (text or multimodal).
    Returns (text_form, blocks_form, is_multimodal).
    """
    formatter = get_formatter(strategy)
    load_images = strategy == ExtractionStrategy.IMAGE

    contexts: List[ItemContext] = []
    for item in items:
        attrs = item.attributes or {}
        colors = list(attrs.get("color", []))
        structured_attrs = {
            supercat: sorted(attrs[supercat])
            for supercat in REFINEMENT_SUPERCATS
            if supercat in attrs
        }
        contexts.append(ItemContext(
            item_id=item.id,
            category=item.category,
            colors=colors,
            attributes=structured_attrs,
            image_path=catalog.image_paths.get(item.id) if load_images else None,
            bbox=catalog.bboxes.get(item.id) if load_images else None,
        ))

    header = (
        f"Selected items ({len(contexts)}) — user picked these from the current "
        f"retrieval as a relevance signal:"
    )
    text_form = formatter.format_text(header, contexts) if contexts else ""
    blocks_form = formatter.build_blocks(header, contexts) if contexts else []
    is_multimodal = bool(getattr(formatter, "is_multimodal", False))
    return text_form, blocks_form, is_multimodal


def run_relevance_feedback(
    selected_items: List[PreferenceItem],
    comment: str,
    search_state: SearchState,
    catalog: FashionpediaCatalog,
    llm_client: LLMClient,
    strategy: ExtractionStrategy = ExtractionStrategy.ATTRIBUTE,
    chat_history: Optional[List[dict]] = None,
) -> FeedbackResult:
    """
    Calls the relevance-feedback LLM with the selected items + comment +
    current state, and returns a refined query + a short response.
    """
    if not selected_items:
        return FeedbackResult(
            response="",
            refined_query=search_state.current_query,
            raw={},
        )

    text_block, blocks, is_multimodal = _build_selected_items_block(
        selected_items, catalog, strategy
    )

    state_section = f"Current search state:\n{search_state.to_context_str()}\n\n"
    comment_section = (
        f"User comment about the selected items: \"{comment.strip()}\"\n\n"
        if comment and comment.strip()
        else "User comment about the selected items: (none — treat the selected items as soft positive feedback)\n\n"
    )
    instruction = "Produce the refined query and updated state as JSON."

    if is_multimodal:
        current_user_content: Union[str, list] = [
            {"type": "text", "text": state_section},
            *blocks,
            {"type": "text", "text": comment_section + instruction},
        ]
        log_user = "[multimodal content]"
    else:
        current_user_content = (
            f"{state_section}{text_block}\n\n{comment_section}{instruction}"
        )
        log_user = current_user_content

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": current_user_content})

    log.llm_prompt(SYSTEM_PROMPT, log_user)

    raw = ""
    try:
        raw = llm_client.chat(messages)
        log.llm_raw(raw)

        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)

        log.llm_parsed({
            "intent": data.get("intent", "relevance_feedback"),
            "updated_query": data.get("refined_query", ""),
            "positive_constraints": data.get("positive_constraints", []),
            "negative_constraints": data.get("negative_constraints", []),
            "style_tags": data.get("style_tags", []),
            "category": data.get("category", ""),
            "occasion": data.get("occasion", ""),
            "suggestions": [],
            "response": data.get("response", ""),
        })

        response = str(data.get("response", "")).strip()
        refined_query = str(data.get("refined_query", "")).strip()
        if not refined_query:
            refined_query = search_state.current_query
        return FeedbackResult(response=response, refined_query=refined_query, raw=data)

    except (json.JSONDecodeError, Exception) as exc:
        log.llm_fallback(raw, str(exc))
        fallback_text = (
            "I'll refine the search based on your selection."
            if not raw else raw
        )
        return FeedbackResult(
            response=fallback_text,
            refined_query=search_state.current_query,
            raw={},
        )
