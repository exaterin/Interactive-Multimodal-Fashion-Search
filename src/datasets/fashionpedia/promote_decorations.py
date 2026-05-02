"""Promote contained decoration/closure crops into parent garment attributes.

For every annotation whose category supercategory is in `PROMOTE_FROM`, find the
smallest sibling annotation (same image) whose bbox contains at least
`CONTAINMENT_THRESHOLD` of the donor's area and whose category is in a
"recipient" supercategory. The donor's category name is added as a synthetic
attribute on that parent.

Writes `data/fashionpedia/derived_attributes.json`:
    { parent_item_id: { supercategory: [attr_name, ...] } }

The annotation parser merges this on top of the original COCO attributes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


PROMOTE_FROM: Set[str] = {"decorations", "closures"}
RECIPIENT_EXCLUDE: Set[str] = {"decorations", "closures", "garment parts"}
CONTAINMENT_THRESHOLD: float = 0.8

SUPERCAT_TO_ATTR_GROUP: Dict[str, str] = {
    "decorations": "textile finishing, manufacturing techniques",
    "closures": "opening type",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FASHIONPEDIA_DIR = PROJECT_ROOT / "data" / "fashionpedia"
ANNOTATIONS_JSON = FASHIONPEDIA_DIR / "instances_attributes_train2020.json"
OUTPUT_JSON = FASHIONPEDIA_DIR / "derived_attributes.json"


def _containment(donor_bbox: List[float], parent_bbox: List[float]) -> float:
    dx, dy, dw, dh = donor_bbox
    px, py, pw, ph = parent_bbox
    ix1, iy1 = max(dx, px), max(dy, py)
    ix2, iy2 = min(dx + dw, px + pw), min(dy + dh, py + ph)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    donor_area = dw * dh
    if donor_area <= 0:
        return 0.0
    return inter / donor_area


def _attr_name(category_name: str) -> str:
    return f"{category_name}(a)"


def main() -> None:
    with ANNOTATIONS_JSON.open() as f:
        data = json.load(f)

    cat_by_id = {c["id"]: c for c in data["categories"]}

    donor_cat_ids = {
        cid for cid, c in cat_by_id.items() if c["supercategory"] in PROMOTE_FROM
    }
    recipient_cat_ids = {
        cid for cid, c in cat_by_id.items()
        if c["supercategory"] not in RECIPIENT_EXCLUDE
    }

    # Group annotations by image
    by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    derived: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    n_donors = 0
    n_donors_with_parent = 0
    promoted_by_cat: Counter = Counter()
    multi_parent_skips = 0

    for anns in by_image.values():
        donors = [a for a in anns if a["category_id"] in donor_cat_ids]
        candidates = [a for a in anns if a["category_id"] in recipient_cat_ids]
        if not donors:
            continue

        for donor in donors:
            n_donors += 1
            donor_cat_name = cat_by_id[donor["category_id"]]["name"]
            donor_super = cat_by_id[donor["category_id"]]["supercategory"]
            target_group = SUPERCAT_TO_ATTR_GROUP[donor_super]

            # Find candidate parents that contain >= threshold of donor's bbox
            matches: List[Tuple[float, dict]] = []
            for cand in candidates:
                if cand["id"] == donor["id"]:
                    continue
                ratio = _containment(donor["bbox"], cand["bbox"])
                if ratio >= CONTAINMENT_THRESHOLD:
                    cand_area = cand["bbox"][2] * cand["bbox"][3]
                    matches.append((cand_area, cand))

            if not matches:
                continue

            # Pick the smallest-area containing parent (most specific)
            matches.sort(key=lambda x: x[0])
            parent = matches[0][1]
            parent_id = str(parent["id"])

            derived[parent_id][target_group].add(_attr_name(donor_cat_name))
            n_donors_with_parent += 1
            promoted_by_cat[donor_cat_name] += 1
            if len(matches) > 1:
                multi_parent_skips += 1

    out = {
        pid: {group: sorted(names) for group, names in groups.items()}
        for pid, groups in derived.items()
    }
    OUTPUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"Promotion threshold: containment >= {CONTAINMENT_THRESHOLD}")
    print(f"Donor supercategories: {sorted(PROMOTE_FROM)}")
    print(f"  donors total:               {n_donors}")
    print(f"  donors with a parent:       {n_donors_with_parent}")
    print(f"  donors with no parent:      {n_donors - n_donors_with_parent}")
    print(f"  donors with >1 candidate:   {multi_parent_skips} (picked smallest)")
    print(f"  parents enriched:           {len(out)}")
    print("\nPromoted donors per category:")
    for name, n in promoted_by_cat.most_common():
        print(f"  {name:<14} {n}")
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
