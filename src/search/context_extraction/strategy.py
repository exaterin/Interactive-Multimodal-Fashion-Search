from __future__ import annotations

from enum import Enum
from typing import Union

from .attribute import AttributeFormatter
from .description import DescriptionFormatter
from .image import ImageFormatter


class ExtractionStrategy(str, Enum):
    """How catalog and preference items are rendered to the LLM."""
    ATTRIBUTE = "attribute"
    DESCRIPTION = "description"
    IMAGE = "image"


Formatter = Union[AttributeFormatter, DescriptionFormatter, ImageFormatter]


_FORMATTERS = {
    ExtractionStrategy.ATTRIBUTE: AttributeFormatter(),
    ExtractionStrategy.DESCRIPTION: DescriptionFormatter(),
    ExtractionStrategy.IMAGE: ImageFormatter(),
}


def get_formatter(strategy: ExtractionStrategy) -> Formatter:
    return _FORMATTERS[strategy]


def parse_strategy(value: str) -> ExtractionStrategy:
    if value in ExtractionStrategy._value2member_map_:
        return ExtractionStrategy(value)
    return ExtractionStrategy.ATTRIBUTE
