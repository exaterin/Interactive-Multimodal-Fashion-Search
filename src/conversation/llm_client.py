import json
from typing import List, Dict

import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_URL


Message = Dict[str, str]


class LLMClient:
    """
    LLM client for interacting with OpenRouter-compatible chat models.
    """

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> None:

        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_URL

        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _build_payload(self, messages: List[Message]) -> dict:
        """
        Build request payload for LLM API.
        """

        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "reasoning": {"enabled": True},
        }

    def chat(self, messages: List[Message]) -> str:
        """
        Send chat history to LLM and return assistant reply.
        """

        payload = self._build_payload(messages)

        response = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Unexpected API response: {data}")

        if not content:
            return "Sorry, the model returned an empty response."

        return content