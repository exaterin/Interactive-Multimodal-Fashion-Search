from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.data.attribute_maps import (
    FABRIC_ATTRIBUTE_MAP,
    FABRIC_PART_NAMES,
    PATTERN_ATTRIBUTE_MAP,
    PATTERN_PART_NAMES,
    SHAPE_ATTRIBUTE_MAP,
    SHAPE_PART_NAMES,
)


def _read_lines(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_fabric_annotations(file_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Expected format:
    <img_name> <upper_fabric> <lower_fabric> <outer_fabric>
    """
    parsed: Dict[str, Dict[str, str]] = {}

    for line_number, line in enumerate(_read_lines(file_path), start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Invalid fabric annotation at line {line_number} in {file_path.name}: {line}"
            )

        image_id = parts[0]
        values = [int(x) for x in parts[1:]]

        parsed[image_id] = {
            part_name: FABRIC_ATTRIBUTE_MAP[value]
            for part_name, value in zip(FABRIC_PART_NAMES, values)
        }

    return parsed


def parse_pattern_annotations(file_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Expected format:
    <img_name> <upper_pattern> <lower_pattern> <outer_pattern>
    """
    parsed: Dict[str, Dict[str, str]] = {}

    for line_number, line in enumerate(_read_lines(file_path), start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Invalid pattern annotation at line {line_number} in {file_path.name}: {line}"
            )

        image_id = parts[0]
        values = [int(x) for x in parts[1:]]

        parsed[image_id] = {
            part_name: PATTERN_ATTRIBUTE_MAP[value]
            for part_name, value in zip(PATTERN_PART_NAMES, values)
        }

    return parsed


def parse_shape_annotations(file_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Expected format:
    <img_name> <shape_0> <shape_1> ... <shape_11>
    """
    parsed: Dict[str, Dict[str, str]] = {}

    expected_length = 1 + len(SHAPE_PART_NAMES)

    for line_number, line in enumerate(_read_lines(file_path), start=1):
        parts = line.split()
        if len(parts) != expected_length:
            raise ValueError(
                f"Invalid shape annotation at line {line_number} in {file_path.name}: {line}"
            )

        image_id = parts[0]
        values = [int(x) for x in parts[1:]]

        parsed[image_id] = {
            part_name: SHAPE_ATTRIBUTE_MAP[idx][value]
            for idx, (part_name, value) in enumerate(zip(SHAPE_PART_NAMES, values))
        }

    return parsed


def build_filter_values(
    fabric_annotations: Dict[str, Dict[str, str]],
    pattern_annotations: Dict[str, Dict[str, str]],
    shape_annotations: Dict[str, Dict[str, str]],
    skip_na: bool = True,
) -> Dict[str, List[str]]:
    """
    Collect unique values for sidebar filters.
    """
    def collect_values(annotations: Dict[str, Dict[str, str]]) -> List[str]:
        values = set()
        for item in annotations.values():
            for value in item.values():
                if skip_na and value == "NA":
                    continue
                values.add(value)
        return sorted(values)

    return {
        "fabric": collect_values(fabric_annotations),
        "pattern": collect_values(pattern_annotations),
        "shape": collect_values(shape_annotations),
    }