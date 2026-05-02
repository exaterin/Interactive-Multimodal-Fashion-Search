"""Drop crops whose Fashionpedia category belongs to an unwanted supercategory.

Filters `fashionpedia_embeddings.npy` and `fashionpedia_item_ids.npy` in place.
Originals are backed up to `*.bak.npy` on the first run.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


DROP_SUPERCATEGORIES = {"garment parts", "legs and feet", "decorations", "closures"}

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

    item_ids = np.load(ITEM_IDS_PATH, allow_pickle=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(item_ids) == len(embeddings), "embedding/item_id length mismatch"

    keep_mask = np.array(
        [category_of_item.get(str(i), -1) not in drop_category_ids for i in item_ids],
        dtype=bool,
    )

    dropped = (~keep_mask).sum()
    kept = keep_mask.sum()

    dropped_counts = Counter(
        category_of_item.get(str(i), -1) for i in item_ids[~keep_mask]
    )

    print(f"Dropping supercategories: {sorted(DROP_SUPERCATEGORIES)}")
    print(f"  matched category ids: {sorted(drop_category_ids)}")
    print(f"Before: {len(item_ids)} crops")
    print(f"After:  {kept} crops  ({dropped} dropped)")
    print("\nDropped per category:")
    for cid, n in sorted(dropped_counts.items(), key=lambda kv: -kv[1]):
        name = category_by_id[cid]["name"]
        print(f"  {name:<32} (id={cid:>2}): {n}")

    print("\nWriting filtered files...")
    _backup_once(EMBEDDINGS_PATH)
    _backup_once(ITEM_IDS_PATH)
    np.save(EMBEDDINGS_PATH, embeddings[keep_mask])
    np.save(ITEM_IDS_PATH, item_ids[keep_mask])
    print("Done.")


if __name__ == "__main__":
    main()
