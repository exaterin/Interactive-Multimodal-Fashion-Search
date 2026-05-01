from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from fashion_clip.fashion_clip import FashionCLIP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.fashionpedia.annotation_parser import parse_fashionpedia_annotations  # noqa: E402

ANNOTATIONS_JSON = PROJECT_ROOT / "data" / "fashionpedia" / "instances_attributes_train2020.json"
EMBEDDINGS_PATH  = PROJECT_ROOT / "data" / "fashionpedia" / "embeddings" / "fashionpedia_embeddings.npy"
ITEM_IDS_PATH    = PROJECT_ROOT / "data" / "fashionpedia" / "embeddings" / "fashionpedia_item_ids.npy"
OUTPUT_PATH      = PROJECT_ROOT / "data" / "fashionpedia" / "color_ann.json"


_COLOR_VOCAB: list[tuple[str, str]] = [
    # neutrals
    ("black",      "black"),
    ("white",      "white"),
    ("gray",       "gray"),
    ("beige",      "beige or cream"),
    ("brown",      "brown"),
    # warm
    ("red",        "red"),
    ("orange",     "orange"),
    ("yellow",     "yellow"),
    ("pink",       "pink"),
    ("coral",      "coral or salmon colored"),
    ("burgundy",   "burgundy or wine colored"),
    # cool
    ("navy",       "navy blue"),
    ("blue",       "blue"),
    ("light_blue", "light blue or sky blue"),
    ("green",      "green"),
    ("olive",      "olive or khaki colored"),
    ("purple",     "purple or violet"),
    ("lavender",   "lavender or lilac colored"),
    ("teal",       "teal or turquoise colored"),
    # multi / special
    ("multicolor", "multicolor or colorful"),
    ("camouflage", "camouflage patterned"),
    ("colorblock", "color block"),
    ("denim",      "denim blue"),
    ("gold",       "gold or metallic golden"),
    ("silver",     "silver or metallic silver"),
]

COLORS  = [(label, f"a photo of a {desc} clothing item") for label, desc in _COLOR_VOCAB]
LABELS  = [c[0] for c in COLORS]
PROMPTS = [c[1] for c in COLORS]


# Helpers

def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def load_item_embeddings() -> tuple[list[str], np.ndarray]:
    """Load and L2-normalise the pre-computed per-crop embeddings."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_PATH}\n"
            "Run: python src/models/generate_fashionpedia_embeddings.py"
        )
    embeddings = normalize(np.load(EMBEDDINGS_PATH).astype(np.float32))
    item_ids   = [str(x) for x in np.load(ITEM_IDS_PATH, allow_pickle=True).tolist()]

    if len(item_ids) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(item_ids)} item IDs vs {len(embeddings)} embeddings"
        )
    return item_ids, embeddings


def encode_color_prompts(model: FashionCLIP) -> np.ndarray:
    """Return (n_colors, D) L2-normalised text embeddings for all color prompts."""
    raw = np.array(model.encode_text(PROMPTS, batch_size=len(PROMPTS)), dtype=np.float32)
    return normalize(raw)


def build_annotations(
    item_ids: list[str],
    similarities: np.ndarray,   # (N, n_colors), cosine similarity in [-1, 1]
    top_k: int,
    threshold: float,
) -> dict[str, dict]:
    """Convert cosine similarity matrix to {item_id: {colors, scores}} dict.

    top_k colors are selected and ranked by raw cosine similarity score.
    Only colors whose score meets `threshold` are kept; an item may have
    fewer than top_k entries if scores fall below the threshold.
    """
    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]  # (N, top_k)

    results: dict[str, dict] = {}
    for i, item_id in enumerate(item_ids):
        colors, scores = [], {}
        for idx in top_indices[i]:
            score = float(similarities[i, idx])
            if score < threshold:
                break                   # indices are sorted desc; no need to continue
            label = LABELS[idx]
            colors.append(label)
            scores[label] = round(score, 4)
        results[item_id] = {"colors": colors, "scores": scores}
    return results


def print_summary(results: dict[str, dict]) -> None:
    counter: Counter = Counter(
        color for v in results.values() for color in v["colors"]
    )
    print("\nTop-15 most frequent colors:")
    for color, count in counter.most_common(15):
        print(f"  {color:<15} {count:>7}")

def main(top_k: int, threshold: float) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load pre-computed item embeddings (no image I/O needed)
    print("Loading item embeddings …")
    item_ids, item_embs = load_item_embeddings()
    print(f"  {len(item_ids)} items  |  embedding dim {item_embs.shape[1]}")

    # 2. Verify item IDs exist in annotation JSON
    print("Parsing Fashionpedia annotations …")
    filename_map, _, _, _ = parse_fashionpedia_annotations(ANNOTATIONS_JSON)
    known_ids = set(filename_map)
    missing = sum(1 for iid in item_ids if iid not in known_ids)
    if missing:
        print(f"  Warning: {missing} item IDs not found in annotations (will still be annotated)")

    # 3. Encode color text prompts
    print("Loading FashionCLIP and encoding color prompts …")
    model = FashionCLIP("fashion-clip")
    model.model = model.model.to(device)
    color_embs = encode_color_prompts(model)            # (n_colors, D)
    print(f"  {len(LABELS)} color prompts encoded")

    # 4. Compute all similarities in one matrix multiply  (N, n_colors)
    print("Computing cosine similarities …")
    similarities = item_embs @ color_embs.T             # (N, n_colors)

    # 5. Build annotation dict
    print(f"Building annotations (top_k={top_k}, cosine threshold={threshold}) …")
    results = build_annotations(item_ids, similarities, top_k, threshold)

    # 6. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} annotations → {OUTPUT_PATH}")

    print_summary(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate Fashionpedia items with colors via FashionCLIP")
    p.add_argument("--top_k",     type=int,   default=2,    help="Colors per item (default: 2)")
    p.add_argument("--threshold", type=float, default=0.25, help="Min cosine similarity to include a color (default: 0.25)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(top_k=args.top_k, threshold=args.threshold)
