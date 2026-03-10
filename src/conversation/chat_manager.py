from typing import Dict, List

from src.conversation.llm_client import LLMClient


Message = Dict[str, str]


class ChatManager:
    """
    Handles conversation logic:
    - stores system prompt
    - trims history before sending it to the model
    - builds final message list for the LLM
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        max_history_messages: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages

    def _trim_history(self, history: List[Message]) -> List[Message]:
        """
        Keep only the last N messages from history.
        """
        return history[-self.max_history_messages :]

    def build_messages(self, history: List[Message]) -> List[Message]:
        """
        Build the final message list sent to the LLM.
        """
        trimmed_history = self._trim_history(history)

        return [
            {"role": "system", "content": self.system_prompt},
            *trimmed_history,
        ]

    def generate_reply(self, history: List[Message]) -> str:
        """
        Generate assistant reply from conversation history.
        """
        messages = self.build_messages(history)
        return self.llm_client.chat(messages)