"""Drop crops whose Fashionpedia category belongs to an unwanted supercategory
or whose attribute count is below `MIN_ATTRIBUTES`.

Filters `fashionpedia_embeddings.npy` and `fashionpedia_item_ids.npy` in place.
Originals are backed up to `*.bak.npy` on the first run.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.datasets.fashionpedia.annotation_parser import (
    parse_fashionpedia_annotations,
)


DROP_SUPERCATEGORIES = {
    "garment parts", "legs and feet", "decorations", "closures",
    "head", "neck", "arms and hands", "others", "waist",
}
MIN_ATTRIBUTES = 3   # drop crops with fewer than this many attributes (color excluded)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FASHIONPEDIA_DIR = PROJECT_ROOT / "data" / "fashionpedia"
ANNOTATIONS_JSON = FASHIONPEDIA_DIR / "instances_attributes_train2020.json"
EMBEDDINGS_PATH = FASHIONPEDIA_DIR / "embeddings" / "fashionpedia_embeddings.npy"
ITEM_IDS_PATH = FASHIONPEDIA_DIR / "embeddings" / "fashionpedia_item_ids.npy"


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(".bak.npy")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backed up {path.name} -> {backup.name}")


def _count_attrs(attr_map_entry: dict) -> int:
    return sum(len(v) for v in attr_map_entry.values())


def main() -> None:
    with ANNOTATIONS_JSON.open() as f:
        ann_data = json.load(f)

    drop_category_ids = {
        c["id"] for c in ann_data["categories"]
        if c["supercategory"] in DROP_SUPERCATEGORIES
    }
    category_by_id = {c["id"]: c for c in ann_data["categories"]}
    category_of_item = {
        str(a["id"]): a["category_id"] for a in ann_data["annotations"]
    }

    _, _, _, attr_map = parse_fashionpedia_annotations(ANNOTATIONS_JSON)

    item_ids = np.load(ITEM_IDS_PATH, allow_pickle=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(item_ids) == len(embeddings), "embedding/item_id length mismatch"

    keep_super = np.array(
        [category_of_item.get(str(i), -1) not in drop_category_ids for i in item_ids],
        dtype=bool,
    )
    keep_attrs = np.array(
        [_count_attrs(attr_map.get(str(i), {})) >= MIN_ATTRIBUTES for i in item_ids],
        dtype=bool,
    )
    keep_mask = keep_super & keep_attrs

    dropped_by_super = (~keep_super).sum()
    dropped_by_attrs = (keep_super & ~keep_attrs).sum()
    kept = keep_mask.sum()

    dropped_super_counts = Counter(
        category_of_item.get(str(i), -1) for i in item_ids[~keep_super]
    )
    dropped_attr_counts = Counter(
        category_of_item.get(str(i), -1) for i in item_ids[keep_super & ~keep_attrs]
    )

    print(f"Dropping supercategories: {sorted(DROP_SUPERCATEGORIES)}")
    print(f"Min attributes (excl. color): {MIN_ATTRIBUTES}")
    print(f"Before: {len(item_ids)} crops")
    print(f"After:  {kept} crops"
          f"  (-{dropped_by_super} by category, -{dropped_by_attrs} by attribute count)")

    if dropped_super_counts:
        print("\nDropped by supercategory rule, per category:")
        for cid, n in sorted(dropped_super_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {category_by_id[cid]['name']:<32} (id={cid:>2}): {n}")
    if dropped_attr_counts:
        print(f"\nDropped by attribute-count < {MIN_ATTRIBUTES}, per category:")
        for cid, n in sorted(dropped_attr_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {category_by_id[cid]['name']:<32} (id={cid:>2}): {n}")

    print("\nWriting filtered files...")
    _backup_once(EMBEDDINGS_PATH)
    _backup_once(ITEM_IDS_PATH)
    np.save(EMBEDDINGS_PATH, embeddings[keep_mask])
    np.save(ITEM_IDS_PATH, item_ids[keep_mask])
    print("Done.")


if __name__ == "__main__":
    main()
