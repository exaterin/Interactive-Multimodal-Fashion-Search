"""Promote contained crops into synthetic attributes on their parent garment.

For every annotation whose category supercategory is a key in
`SUPERCAT_TO_ATTR_GROUP`, find the smallest sibling annotation in the same
image whose bbox contains at least `CONTAINMENT_THRESHOLD` of the donor's
area and whose category is in a "recipient" supercategory. The donor's
category name (with an `(a)` suffix) is added as a synthetic attribute on
that parent under the mapped attribute supercategory.

Writes `data/fashionpedia/derived_attributes.json`:
    { parent_item_id: { supercategory: [attr_name, ...] } }

Re-runnable: existing entries in the JSON are merged with the new ones.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Donor supercategory → target attribute supercategory.
# Decoration crops (ruffle, bead, ...) become finishing attributes on the parent.
# Closure crops (zipper, buckle) become opening-type attributes.
# Accessory crops (watch, tie, belt, ...) become accessories attributes.
SUPERCAT_TO_ATTR_GROUP: Dict[str, str] = {
    "decorations":     "textile finishing, manufacturing techniques",
    "closures":        "opening type",
    "head":            "accessories",
    "neck":            "accessories",
    "arms and hands":  "accessories",
    "others":          "accessories",
    "waist":           "accessories",
}

PROMOTE_FROM: Set[str] = set(SUPERCAT_TO_ATTR_GROUP.keys())
# Recipients are any annotation NOT in the donor set and not a "garment parts"
# crop (sleeves/collars/etc., which we drop and which would over-promote).
RECIPIENT_EXCLUDE: Set[str] = PROMOTE_FROM | {"garment parts"}
CONTAINMENT_THRESHOLD: float = 0.8

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

    by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    derived: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    # Pre-seed from any existing file so reruns are idempotent and additive.
    if OUTPUT_JSON.exists():
        for pid, groups in json.loads(OUTPUT_JSON.read_text()).items():
            for sc, names in groups.items():
                derived[pid][sc].update(names)

    promoted_by_cat: Counter = Counter()
    orphans_by_cat: Counter = Counter()

    for anns in by_image.values():
        donors = [a for a in anns if a["category_id"] in donor_cat_ids]
        candidates = [a for a in anns if a["category_id"] in recipient_cat_ids]
        if not donors:
            continue

        for donor in donors:
            donor_cat = cat_by_id[donor["category_id"]]
            target_group = SUPERCAT_TO_ATTR_GROUP[donor_cat["supercategory"]]

            matches: List[Tuple[float, dict]] = []
            for cand in candidates:
                if cand["id"] == donor["id"]:
                    continue
                if _containment(donor["bbox"], cand["bbox"]) >= CONTAINMENT_THRESHOLD:
                    cand_area = cand["bbox"][2] * cand["bbox"][3]
                    matches.append((cand_area, cand))

            if not matches:
                orphans_by_cat[donor_cat["name"]] += 1
                continue

            # Most-specific (smallest) parent wins
            matches.sort(key=lambda x: x[0])
            parent_id = str(matches[0][1]["id"])
            derived[parent_id][target_group].add(_attr_name(donor_cat["name"]))
            promoted_by_cat[donor_cat["name"]] += 1

    out = {
        pid: {group: sorted(names) for group, names in groups.items()}
        for pid, groups in derived.items()
    }
    OUTPUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"Promotion threshold: containment >= {CONTAINMENT_THRESHOLD}")
    print(f"Donor supercategories: {sorted(PROMOTE_FROM)}")
    print(f"Parents enriched (total in JSON): {len(out)}")

    print("\nPromoted donors per category:")
    all_cats = set(promoted_by_cat) | set(orphans_by_cat)
    rows = [(c, promoted_by_cat[c], promoted_by_cat[c] + orphans_by_cat[c]) for c in all_cats]
    rows.sort(key=lambda r: -r[1])
    for name, n, total in rows:
        print(f"  {name:<40} {n:>5}/{total:<5} ({100*n/total:>4.0f}%)")
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
