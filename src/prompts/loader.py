"""Load system prompts from src/prompts/<name>.txt."""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


# Output contract appended after the retrieval prompt + the conversation
# prompt. The two modules write into ONE flat JSON object — the schema below
# is the source of truth, overriding any per-module schema mentioned earlier.
# Still ONE LLM call per pipeline step.
_DUAL_OUTPUT_CONTRACT = """\

============================================================
OUTPUT CONTRACT (read carefully — overrides any earlier schema)
============================================================

You make ONE response. The retrieval module and the conversation module both
write into the SAME flat JSON object. Return ONLY this JSON object — no
markdown fences, no commentary, no extra text. Use these EXACT field names.

Schema:
{
  "response": "string",
  "suggestions": ["string", ...],
  "__QUERY_FIELD__": "string",
  "intent": "string",
  "category": "string or empty",
  "positive_constraints": ["string", ...],
  "negative_constraints": ["string", ...],
  "style_tags": ["string", ...],
  "occasion": "string or empty"
}

Field ownership:
- "response" + "suggestions"            → produced by the conversation module.
- "__QUERY_FIELD__" + "intent" + state  → produced by the retrieval module.

Hard rules:
- ALL fields above MUST be present in every response (use "" or [] when empty).
- Output a SINGLE flat object. Do NOT nest the fields inside "retrieval" or
  "conversation" wrappers. Do NOT use "chat_messages" — use "response".
- "response" is one string (1–3 sentences). "suggestions" is an array of
  short clickable phrases as defined in the conversation module.
- On a reset: response is a tiny acknowledgement, suggestions = [], the
  retrieval fields are empty values.
"""


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def load_dual_prompt(retrieval_name: str, retrieval_query_field: str) -> str:
    """
    Concatenate a retrieval-focused prompt + the shared conversation prompt
    and append the unified flat-schema output contract.

    `retrieval_query_field` is the name of the query field the retrieval
    module emits ("updated_query" for /chat, "refined_query" for /feedback).
    """
    retrieval = load_prompt(retrieval_name)
    conversation = load_prompt("conversation")
    contract = _DUAL_OUTPUT_CONTRACT.replace("__QUERY_FIELD__", retrieval_query_field)
    return f"{retrieval}\n\n--- CONVERSATION MODULE ---\n\n{conversation}\n{contract}"
