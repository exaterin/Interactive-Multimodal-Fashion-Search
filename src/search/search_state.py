from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchState:
    original_query: str = ""
    current_query: str = ""
    category: str = ""
    positive_constraints: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    occasion: str = ""
    budget: str = ""
    last_suggestions: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.original_query = ""
        self.current_query = ""
        self.category = ""
        self.positive_constraints = []
        self.negative_constraints = []
        self.style_tags = []
        self.occasion = ""
        self.budget = ""
        self.last_suggestions = []

    def to_context_str(self) -> str:
        parts = []
        if self.current_query:
            parts.append(f"Current search query: {self.current_query}")
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.positive_constraints:
            parts.append(f"Must have: {', '.join(self.positive_constraints)}")
        if self.negative_constraints:
            parts.append(f"Must NOT have: {', '.join(self.negative_constraints)}")
        if self.style_tags:
            parts.append(f"Style tags: {', '.join(self.style_tags)}")
        if self.occasion:
            parts.append(f"Occasion: {self.occasion}")
        if self.budget:
            parts.append(f"Budget range: {self.budget}")
        return "\n".join(parts) if parts else "No active constraints (first query)"

    def update_from_llm(self, llm_data: dict) -> None:

        if llm_data.get("category"):
            self.category = llm_data["category"]

        if "positive_constraints" in llm_data:
            self.positive_constraints = [c for c in llm_data["positive_constraints"] if c]
        if "negative_constraints" in llm_data:
            self.negative_constraints = [c for c in llm_data["negative_constraints"] if c]
        if "style_tags" in llm_data:
            self.style_tags = [t for t in llm_data["style_tags"] if t]

        if llm_data.get("occasion"):
            self.occasion = llm_data["occasion"]
