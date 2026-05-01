from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

COLOR_SIMILARITY_THRESHOLD: float = 0.3

# Attribute supercategories mapped to each filter group
_PATTERN_SUPERCATS: frozenset[str] = frozenset({"textile pattern", "animal"})
_SHAPE_SUPERCATS: frozenset[str] = frozenset(
    {"silhouette", "neckline type", "length", "waistline", "opening type"}
)
_MATERIAL_SUPERCATS: frozenset[str] = frozenset(
    {"non-textile material type", "leather"}
)
_FINISHING_SUPERCATS: frozenset[str] = frozenset(
    {"textile finishing, manufacturing techniques"}
)


def parse_fashionpedia_annotations(
    json_path: Path,
) -> Tuple[
    Dict[str, str],                            # item_id → image file_name
    Dict[str, List[float]],                    # item_id → [x, y, w, h]
    Dict[str, str],                            # item_id → category name
    Dict[str, Dict[str, Set[str]]],            # item_id → supercat → attr names
]:
    """
    Parse the Fashionpedia COCO-format JSON at the annotation (garment instance) level.

    Each annotation corresponds to one garment bounding box in an image.
    Returns four dicts keyed by item_id (annotation ID as string):

        filename_map  : item_id → image file_name (e.g. "abc123.jpg")
        bbox_map      : item_id → [x, y, width, height] in pixels
        category_map  : item_id → garment category name
        attribute_map : item_id → {supercategory: {attribute_name, …}}
                        (only supercategories in the three filter groups)

    If a sibling `derived_attributes.json` is present next to `json_path`, its
    entries are merged on top of the parsed `attribute_map`. See
    `promote_decorations.py`.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Fashionpedia annotation file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Lookup tables
    id_to_filename: Dict[int, str] = {
        img["id"]: img["file_name"] for img in data["images"]
    }
    id_to_category: Dict[int, str] = {
        cat["id"]: cat["name"] for cat in data["categories"]
    }
    id_to_attr: Dict[int, dict] = {
        attr["id"]: attr for attr in data["attributes"]
    }

    filename_map: Dict[str, str] = {}
    bbox_map: Dict[str, List[float]] = {}
    category_map: Dict[str, str] = {}
    attribute_map: Dict[str, Dict[str, Set[str]]] = {}

    for ann in data["annotations"]:
        item_id = str(ann["id"])
        image_id = ann["image_id"]

        filename_map[item_id] = id_to_filename[image_id]
        bbox_map[item_id] = ann["bbox"]          # [x, y, w, h]
        category_map[item_id] = id_to_category[ann["category_id"]]

        attr_groups: Dict[str, Set[str]] = {}
        for attr_id in ann.get("attribute_ids", []):
            attr = id_to_attr.get(attr_id)
            if attr is None:
                continue
            supercat = attr["supercategory"]
            if supercat in _PATTERN_SUPERCATS | _SHAPE_SUPERCATS | _MATERIAL_SUPERCATS | _FINISHING_SUPERCATS:
                attr_groups.setdefault(supercat, set()).add(attr["name"])

        attribute_map[item_id] = attr_groups

    derived_path = json_path.parent / "derived_attributes.json"
    if derived_path.exists():
        with derived_path.open("r", encoding="utf-8") as f:
            derived: Dict[str, Dict[str, List[str]]] = json.load(f)
        for item_id, groups in derived.items():
            target = attribute_map.setdefault(item_id, {})
            for supercat, names in groups.items():
                target.setdefault(supercat, set()).update(names)

    return filename_map, bbox_map, category_map, attribute_map


def load_color_annotations(
    json_path: Path,
    threshold: float = COLOR_SIMILARITY_THRESHOLD,
) -> Dict[str, List[str]]:
    """Load color_ann.json and return item_id → [color_label, …].

    Only colors whose stored cosine similarity score meets `threshold` are kept.
    Returns an empty dict if the file does not exist (graceful degradation).
    """
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return {
        item_id: [
            label
            for label, score in entry["scores"].items()
            if score >= threshold
        ]
        for item_id, entry in data.items()
    }


def build_filter_values(
    category_map: Dict[str, str],
    attribute_map: Dict[str, Dict[str, Set[str]]],
    color_annotations: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Collect unique values for each filter group across all annotations.

    Returns a dict with keys:
        "category" → sorted list of garment category names
        "pattern"  → sorted list of textile pattern attribute names
        "shape"    → sorted list of silhouette / neckline / length / etc. names
        "material" → sorted list of non-textile and leather material names
    """
    categories: Set[str] = set(category_map.values())
    patterns: Set[str] = set()
    shapes: Set[str] = set()
    materials: Set[str] = set()
    finishings: Set[str] = set()

    for supercat_dict in attribute_map.values():
        for supercat, attr_set in supercat_dict.items():
            if supercat in _PATTERN_SUPERCATS:
                patterns.update(attr_set)
            elif supercat in _SHAPE_SUPERCATS:
                shapes.update(attr_set)
            elif supercat in _MATERIAL_SUPERCATS:
                materials.update(attr_set)
            elif supercat in _FINISHING_SUPERCATS:
                finishings.update(attr_set)

    colors: Set[str] = set()
    if color_annotations:
        for labels in color_annotations.values():
            colors.update(labels)

    return {
        "category": sorted(categories),
        "pattern": sorted(patterns),
        "shape": sorted(shapes),
        "material": sorted(materials),
        "finishing": sorted(finishings),
        "color": sorted(colors),
    }
