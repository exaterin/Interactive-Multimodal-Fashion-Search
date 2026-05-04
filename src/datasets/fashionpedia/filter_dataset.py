"""Drop crops whose Fashionpedia category belongs to an unwanted supercategory
or whose attribute count is below `MIN_ATTRIBUTES`.

Filters every bank under `data/fashionpedia/embeddings/`:
  * the legacy top-level bank   (fashionpedia_embeddings.npy / fashionpedia_item_ids.npy)
  * each per-model subdirectory (embeddings_<model>/) including its filenames.npy

All files are rewritten in place. The first time each file is filtered, the
original is copied next to it as `*.bak.npy`.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional

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
EMBEDDINGS_DIR   = FASHIONPEDIA_DIR / "embeddings"


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(".bak.npy")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"    backed up {path.name} -> {backup.name}")


def _count_attrs(attr_map_entry: dict) -> int:
    return sum(len(v) for v in attr_map_entry.values())


def _compute_keep_mask(
    item_ids: np.ndarray,
    category_of_item: dict,
    drop_category_ids: set,
    attr_map: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (combined_mask, keep_super_mask, keep_attrs_mask)."""
    keep_super = np.array(
        [category_of_item.get(str(i), -1) not in drop_category_ids for i in item_ids],
        dtype=bool,
    )
    keep_attrs = np.array(
        [_count_attrs(attr_map.get(str(i), {})) >= MIN_ATTRIBUTES for i in item_ids],
        dtype=bool,
    )
    return keep_super & keep_attrs, keep_super, keep_attrs


def _filter_bank(
    label: str,
    emb_path: Path,
    ids_path: Path,
    filenames_path: Optional[Path],
    category_of_item: dict,
    drop_category_ids: set,
    attr_map: dict,
) -> Optional[tuple[int, int]]:
    """Filter a single bank in place. Returns (before, after) or None if skipped."""
    if not emb_path.exists() or not ids_path.exists():
        print(f"  [{label}] skipped — missing files")
        return None

    item_ids = np.load(ids_path, allow_pickle=True)
    embeddings = np.load(emb_path)
    if len(item_ids) != len(embeddings):
        raise RuntimeError(
            f"[{label}] embedding/item_id length mismatch "
            f"({len(embeddings)} vs {len(item_ids)})"
        )

    filenames = None
    if filenames_path is not None and filenames_path.exists():
        filenames = np.load(filenames_path, allow_pickle=True)
        if len(filenames) != len(item_ids):
            raise RuntimeError(
                f"[{label}] filenames length mismatch "
                f"({len(filenames)} vs {len(item_ids)})"
            )

    keep_mask, _, _ = _compute_keep_mask(
        item_ids, category_of_item, drop_category_ids, attr_map
    )
    kept = int(keep_mask.sum())
    before = len(item_ids)

    if kept == before:
        print(f"  [{label}] already filtered ({before} crops) — nothing to do")
        return before, before

    print(f"  [{label}] {before} -> {kept} crops")
    _backup_once(emb_path)
    _backup_once(ids_path)
    np.save(emb_path, embeddings[keep_mask])
    np.save(ids_path, item_ids[keep_mask])
    if filenames is not None:
        _backup_once(filenames_path)
        np.save(filenames_path, filenames[keep_mask])
    return before, kept


def _discover_targets() -> List[tuple[str, Path, Path, Optional[Path]]]:
    """Find every bank to filter: legacy top-level + each embeddings_*/ subdir."""
    targets: List[tuple[str, Path, Path, Optional[Path]]] = []

    legacy_emb = EMBEDDINGS_DIR / "fashionpedia_embeddings.npy"
    legacy_ids = EMBEDDINGS_DIR / "fashionpedia_item_ids.npy"
    if legacy_emb.exists():
        targets.append(("top-level", legacy_emb, legacy_ids, None))

    for sub in sorted(EMBEDDINGS_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("embeddings_"):
            continue
        targets.append((
            sub.name,
            sub / "fashionpedia_embeddings.npy",
            sub / "fashionpedia_item_ids.npy",
            sub / "fashionpedia_filenames.npy",
        ))
    return targets


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

    targets = _discover_targets()
    if not targets:
        print(f"No banks found under {EMBEDDINGS_DIR}")
        return

    print(f"Dropping supercategories: {sorted(DROP_SUPERCATEGORIES)}")
    print(f"Min attributes (excl. color): {MIN_ATTRIBUTES}")
    print(f"Targets ({len(targets)}):")
    for label, *_ in targets:
        print(f"  - {label}")

    # Report stats once against the first non-filtered bank so the per-category
    # breakdown isn't repeated for every model.
    reported = False
    for label, emb_path, ids_path, _ in targets:
        if not emb_path.exists():
            continue
        item_ids = np.load(ids_path, allow_pickle=True)
        keep_mask, keep_super, keep_attrs = _compute_keep_mask(
            item_ids, category_of_item, drop_category_ids, attr_map
        )
        if keep_mask.sum() == len(item_ids):
            continue  # already filtered; try the next one for the stats
        dropped_super_counts = Counter(
            category_of_item.get(str(i), -1) for i in item_ids[~keep_super]
        )
        dropped_attr_counts = Counter(
            category_of_item.get(str(i), -1) for i in item_ids[keep_super & ~keep_attrs]
        )
        print(f"\nStats from [{label}]:")
        print(f"  Before: {len(item_ids)} crops")
        print(f"  After:  {int(keep_mask.sum())} crops"
              f"  (-{int((~keep_super).sum())} by category,"
              f" -{int((keep_super & ~keep_attrs).sum())} by attribute count)")
        if dropped_super_counts:
            print("  Dropped by supercategory rule, per category:")
            for cid, n in sorted(dropped_super_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {category_by_id[cid]['name']:<32} (id={cid:>2}): {n}")
        if dropped_attr_counts:
            print(f"  Dropped by attribute-count < {MIN_ATTRIBUTES}, per category:")
            for cid, n in sorted(dropped_attr_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {category_by_id[cid]['name']:<32} (id={cid:>2}): {n}")
        reported = True
        break
    if not reported:
        print("\nAll banks already filtered — nothing to drop.")

    print("\nWriting filtered files...")
    for label, emb_path, ids_path, filenames_path in targets:
        _filter_bank(
            label, emb_path, ids_path, filenames_path,
            category_of_item, drop_category_ids, attr_map,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
