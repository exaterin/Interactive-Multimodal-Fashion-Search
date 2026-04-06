from __future__ import annotations

import json
import re
from typing import List, Tuple

from src.conversation.llm_client import LLMClient
from src.search.grounding_analyzer import GroundingContext
from src.search.search_state import SearchState


_SYSTEM_PROMPT = """\
You are a grounded fashion search assistant helping users explore a clothing catalog.

You have access to the ACTUAL results retrieved from the catalog for the current query.
Your job is to:
1. Understand the user's intent (initial search, refinement, constraint change, reset, etc.)
2. Write a SHORT, helpful response (1-3 sentences) that references what was actually found.
3. Propose 2-4 SPECIFIC, clickable refinement suggestions based ONLY on attributes/categories
   that actually appear in the retrieved results.
4. Produce an updated search query to use for the next retrieval.
5. Optionally extract structured constraints from the user message.

Intent types:
- initial_search        — first query about a new item
- initial_specific      — first query already very specific (no clarification needed)
- positive_refinement   — "more elegant", "I like floral", "show me fitted ones"
- negative_refinement   — "not leather", "without floral", "less formal"
- add_constraint        — "also in blue", "with short sleeves"
- remove_constraint     — "remove the color filter", "forget the length"
- style_or_occasion     — "for a wedding", "casual summer look"
- browse_intent         — "show me more", "what else is there"
- reset                 — "start over", "new search", "forget everything"

Rules:
- Suggestions MUST be grounded in what the results actually contain. Never invent attributes.
- For broad results with many categories: suggest narrowing by category or style.
- For specific results: suggest small variations (length, color, pattern).
- Do NOT ask vague questions like "what style do you prefer?".
- If the query is already very specific and results look good, use fewer suggestions.
- Keep the response warm but brief.
- The updated_query must be a concise retrieval phrase (not a sentence).
- For a reset intent: set updated_query to "" and suggestions to [].

Return ONLY a valid JSON object. Do not add markdown fences or extra text.

Schema:
{
  "response": "string",
  "suggestions": ["string", ...],
  "updated_query": "string",
  "intent": "string",
  "category": "string or empty",
  "positive_constraints": ["string", ...],
  "negative_constraints": ["string", ...],
  "style_tags": ["string", ...],
  "occasion": "string or empty"
}
"""


def generate_grounded_response(
    user_message: str,
    search_state: SearchState,
    grounding_context: GroundingContext,
    llm_client: LLMClient,
) -> Tuple[str, List[str], str, dict]:
    """
    Generate a grounded assistant response using actual retrieval context.

    Returns:
        (response_text, suggestions, updated_query, raw_llm_data)
    """
    user_content = (
        f"Current search state:\n{search_state.to_context_str()}\n\n"
        f"What was actually retrieved from the catalog:\n"
        f"{grounding_context.to_prompt_str()}\n\n"
        f"User's message: \"{user_message}\"\n\n"
        f"Respond using ONLY information present in the retrieved results above."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    try:
        raw = llm_client.chat(messages)
        # Strip markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(cleaned)

        response = data.get("response", "").strip()
        suggestions = [s for s in data.get("suggestions", []) if s][:4]
        updated_query = data.get("updated_query", search_state.current_query).strip()

        return response, suggestions, updated_query, data

    except (json.JSONDecodeError, Exception):
        # Graceful fallback: show raw reply, no suggestions
        fallback_text = raw if raw else "I found some results. Let me know how you'd like to refine."
        return fallback_text, [], search_state.current_query, {}
