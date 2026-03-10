import requests
import json
from llm_prompt import SYSTEM_PROMPT

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def parse_with_llm(messages: list[dict]) -> str:
    """
    Send chat history to the LLM and return assistant response.

    Args:
        messages: list of dicts like
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]

    Returns:
        Assistant reply as string.
    """
    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": "google/gemini-3-flash-preview",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages,
                ],
                "temperature": 0.0,
                "reasoning": {"enabled": True},
            }
        ),
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    return content