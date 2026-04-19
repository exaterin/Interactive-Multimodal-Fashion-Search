"""
Experiment: CLIP retrieval + LLM analysis of top-10 image results

For each fashion query, retrieves top-10 items with FashionCLIP,
sends them to google/gemini-2.0-flash-001 via OpenRouter asking what is
common from a fashion catalog perspective. Save:
  - PNG visualization: 10-image grid with attributes + LLM answer
  - JSON: full attributes for each retrieved item + LLM response
"""
from __future__ import annotations

import base64
import io
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OPENROUTER_API_KEY, OPENROUTER_URL
from src.data.fashionpedia.annotation_parser import (
    build_filter_values,
    load_color_annotations,
    parse_fashionpedia_annotations,
)
from src.data.fashionpedia.catalog import FashionpediaCatalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder

# ── Config ────────────────────────────────────────────────────────────────────

QUERIES = [
    "beach dress",
    "formal evening dress",
    "summer casual outfit",
    "winter coat outfit",
    "business formal clothes",
    "minimalist black outfit",
    "elegant dinner outfit",
    "streetwear outfit",
    "sporty casual look",
    "old money style outfit",
    "clean girl aesthetic outfit",
    "soft feminine outfit",
    "edgy fashion look",
    "vintage inspired outfit",
    "luxury chic outfit",
    "outfit for a summer date",
    "what to wear to a beach party",
    "elegant but comfortable outfit",
    "outfit for a job interview",
    "vacation outfit ideas",
]

TOP_K = 10
THUMBNAIL_SIZE = (200, 200)
LLM_MODEL = "google/gemini-2.0-flash-001"
GRID_COLS = 5
GRID_ROWS = 2  # 10 images total

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "clip_llm_analysis"

FASHIONPEDIA_DIR = PROJECT_ROOT / "data" / "fashionpedia"
ANNOTATIONS_JSON = FASHIONPEDIA_DIR / "instances_attributes_train2020.json"
IMAGES_DIR = FASHIONPEDIA_DIR / "train"
EMBEDDINGS_DIR = FASHIONPEDIA_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "fashionpedia_embeddings.npy"
ITEM_IDS_PATH = EMBEDDINGS_DIR / "fashionpedia_item_ids.npy"
COLOR_ANN_PATH = FASHIONPEDIA_DIR / "color_ann.json"

LLM_PROMPT = (
    "You are a fashion expert and catalog analyst. "
    "Below are 10 items retrieved from a fashion catalog in response to a search query. "
    "Describe what these items have in common from a fashion catalog perspective: "
    "consider garment type, silhouette, textile patterns, colors, styling details, "
    "and any recurring visual themes. "
    "Be specific and concise — 3-5 sentences."
)

# Supercategory → short display label for visualization
SUPERCAT_LABELS: Dict[str, str] = {
    "textile pattern": "pattern",
    "animal": "pattern",
    "silhouette": "silhouette",
    "neckline type": "neckline",
    "length": "length",
    "waistline": "waist",
    "opening type": "opening",
    "non-textile material type": "material",
    "leather": "material",
    "textile finishing, manufacturing techniques": "finish",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def load_catalog() -> FashionpediaCatalog:
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    embeddings = _normalize_rows(embeddings)
    item_ids = [str(x) for x in np.load(ITEM_IDS_PATH, allow_pickle=True).tolist()]

    print("Parsing annotations...")
    filename_map, bbox_map, category_map, attribute_map = parse_fashionpedia_annotations(
        ANNOTATIONS_JSON
    )

    embedded_set = set(item_ids)
    filename_map  = {k: v for k, v in filename_map.items()  if k in embedded_set}
    bbox_map      = {k: v for k, v in bbox_map.items()      if k in embedded_set}
    category_map  = {k: v for k, v in category_map.items()  if k in embedded_set}
    attribute_map = {k: v for k, v in attribute_map.items() if k in embedded_set}

    color_annotations = {
        k: v
        for k, v in load_color_annotations(COLOR_ANN_PATH).items()
        if k in embedded_set
    }
    available_values = build_filter_values(category_map, attribute_map, color_annotations)
    image_paths = {
        item_id: IMAGES_DIR / fname
        for item_id, fname in filename_map.items()
    }

    return FashionpediaCatalog(
        item_ids=item_ids,
        embeddings=embeddings,
        image_paths=image_paths,
        bboxes=bbox_map,
        category_annotations=category_map,
        attribute_annotations=attribute_map,
        color_annotations=color_annotations,
        available_values=available_values,
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query: str,
    catalog: FashionpediaCatalog,
    encoder: FashionCLIPEncoder,
    top_k: int,
) -> List[dict]:
    query_emb = encoder.encode_text(query)
    scores = catalog.embeddings @ query_emb
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        item_id = catalog.item_ids[idx]
        attrs_raw = catalog.attribute_annotations.get(item_id, {})

        results.append({
            "rank": rank,
            "item_id": item_id,
            "score": float(scores[idx]),
            "image_path": catalog.image_paths.get(item_id),
            "bbox": catalog.bboxes.get(item_id, [0, 0, 64, 64]),
            "category": catalog.category_annotations.get(item_id, "unknown"),
            "colors": catalog.color_annotations.get(item_id, []),
            "attributes": {
                supercat: sorted(attrs)
                for supercat, attrs in attrs_raw.items()
                if attrs
            },
        })
    return results


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_crop(image_path: Path, bbox: list) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path).convert("RGB")
        x, y, w, h = (max(0, int(v)) for v in bbox)
        w, h = max(1, w), max(1, h)
        crop = img.crop((x, y, x + w, y + h))
        return crop.resize(THUMBNAIL_SIZE, Image.LANCZOS)
    except Exception as e:
        print(f"    Warning: could not load {image_path}: {e}")
        return None


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm_with_images(crops: List[Image.Image], prompt: str) -> str:
    content = [{"type": "text", "text": prompt}]
    for crop in crops:
        b64 = pil_to_base64(crop)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
    }
    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=120,
    )
    if not response.ok:
        raise ValueError(f"API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"Unexpected API response: {data}")


# ── Attribute formatting ───────────────────────────────────────────────────────

def format_attr_caption(result: dict) -> str:
    """Build a compact multi-line attribute caption for one image cell."""
    lines = []

    category = result["category"]
    score = result["score"]
    lines.append(f"#{result['rank']}  {category}  ({score:.3f})")

    colors = result["colors"]
    if colors:
        lines.append("color: " + ", ".join(colors[:3]))

    # Group supercategories by display label to avoid redundancy
    grouped: Dict[str, List[str]] = {}
    for supercat, vals in result["attributes"].items():
        label = SUPERCAT_LABELS.get(supercat, supercat)
        grouped.setdefault(label, []).extend(vals)

    # Priority order for display
    for label in ["pattern", "silhouette", "length", "neckline", "material", "finish", "waist", "opening"]:
        vals = grouped.get(label)
        if vals:
            val_str = ", ".join(sorted(set(vals)))
            # Truncate long values
            if len(val_str) > 28:
                val_str = val_str[:25] + "…"
            lines.append(f"{label}: {val_str}")

    return "\n".join(lines)


# ── Visualization ─────────────────────────────────────────────────────────────

def save_visualization(
    query: str,
    results: List[dict],
    crops: List[Optional[Image.Image]],
    llm_response: str,
    output_path: Path,
) -> None:
    n = len(crops)
    # Image rows + 1 attribute-text row per image row + 1 LLM response row
    # Layout: for each image row, we pair it with a thin text row below
    # Total subplot rows: GRID_ROWS * 2 (image + attr) + 1 (LLM)
    n_img_rows = GRID_ROWS
    n_total_rows = n_img_rows * 2 + 1

    img_h = 2.5      # height per image row
    attr_h = 1.2     # height per attribute text row
    llm_h = 2.8      # height for LLM response
    col_w = 3.2      # width per column

    fig_w = GRID_COLS * col_w
    fig_h = n_img_rows * (img_h + attr_h) + llm_h + 0.6  # +0.6 for title

    height_ratios = []
    for _ in range(n_img_rows):
        height_ratios.extend([img_h, attr_h])
    height_ratios.append(llm_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_total_rows, GRID_COLS,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.06,
    )

    fig.suptitle(
        f'Query: "{query}"  —  Top {n} CLIP Results  |  {LLM_MODEL}',
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    # ── Image cells and attribute cells ───────────────────────────────────────
    for i, (result, crop) in enumerate(zip(results, crops)):
        img_row = (i // GRID_COLS) * 2      # 0 or 2
        attr_row = img_row + 1               # 1 or 3
        col = i % GRID_COLS

        # Image
        ax_img = fig.add_subplot(gs[img_row, col])
        if crop is not None:
            ax_img.imshow(np.array(crop))
        else:
            ax_img.set_facecolor("#d0d0d0")
            ax_img.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax_img.transAxes, fontsize=10, color="#555")
        ax_img.set_title(
            f"#{result['rank']}  score {result['score']:.3f}",
            fontsize=7, pad=2, color="#333",
        )
        ax_img.axis("off")

        # Attributes
        ax_attr = fig.add_subplot(gs[attr_row, col])
        ax_attr.axis("off")

        caption_lines = []
        caption_lines.append(result["category"])

        colors = result["colors"]
        if colors:
            caption_lines.append("🎨 " + ", ".join(colors[:3]))

        grouped: Dict[str, List[str]] = {}
        for supercat, vals in result["attributes"].items():
            label = SUPERCAT_LABELS.get(supercat, supercat)
            grouped.setdefault(label, []).extend(vals)

        for label in ["pattern", "silhouette", "length", "neckline", "material"]:
            vals = grouped.get(label)
            if vals:
                val_str = ", ".join(sorted(set(vals)))
                if len(val_str) > 22:
                    val_str = val_str[:19] + "…"
                caption_lines.append(f"{label}: {val_str}")

        caption_text = "\n".join(caption_lines)
        ax_attr.text(
            0.5, 0.98,
            caption_text,
            transform=ax_attr.transAxes,
            fontsize=6.5,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#f7f7f7",
                edgecolor="#ccc",
                linewidth=0.6,
            ),
            linespacing=1.4,
        )

    # Hide unused cells (if n < GRID_ROWS * GRID_COLS)
    for i in range(n, GRID_ROWS * GRID_COLS):
        img_row = (i // GRID_COLS) * 2
        attr_row = img_row + 1
        col = i % GRID_COLS
        for row in (img_row, attr_row):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

    # ── LLM response ─────────────────────────────────────────────────────────
    ax_llm = fig.add_subplot(gs[n_img_rows * 2, :])
    ax_llm.axis("off")

    # Wrap text to fit figure width (~110 chars)
    wrapped = textwrap.fill(llm_response, width=110)
    full_text = f"LLM analysis ({LLM_MODEL}):\n\n{wrapped}"
    ax_llm.text(
        0.01, 0.97,
        full_text,
        transform=ax_llm.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#eef2ff",
            edgecolor="#99aacc",
            linewidth=1,
        ),
        linespacing=1.5,
    )

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved visualization → {output_path.name}")


# ── JSON output ───────────────────────────────────────────────────────────────

def build_result_json(query: str, results: List[dict], llm_response: str) -> dict:
    items = []
    for r in results:
        items.append({
            "rank": r["rank"],
            "item_id": r["item_id"],
            "similarity_score": round(r["score"], 6),
            "category": r["category"],
            "colors": r["colors"],
            "attributes": r["attributes"],
        })
    return {
        "query": query,
        "model": LLM_MODEL,
        "top_k": TOP_K,
        "results": items,
        "llm_analysis": llm_response,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def safe_slug(text: str) -> str:
    return (
        text.lower().strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("'", "")
    )


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FashionCLIP encoder...")
    encoder = FashionCLIPEncoder()

    print("Loading Fashionpedia catalog...")
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog.item_ids)} items\n")

    for qi, query in enumerate(QUERIES, start=1):
        slug = safe_slug(query)
        print(f"[{qi}/{len(QUERIES)}] Query: '{query}'")

        # 1. Retrieve top-K with attributes
        results = retrieve_top_k(query, catalog, encoder, TOP_K)

        # 2. Load crops
        crops: List[Optional[Image.Image]] = [
            load_crop(r["image_path"], r["bbox"]) if r["image_path"] else None
            for r in results
        ]
        valid_crops = [c for c in crops if c is not None]
        print(f"    Images loaded: {len(valid_crops)}/{len(crops)}")

        # 3. LLM analysis
        print(f"    Calling LLM ({LLM_MODEL})...")
        if valid_crops:
            try:
                llm_response = call_llm_with_images(valid_crops, LLM_PROMPT)
                print(f"    Response: {llm_response[:100]}…")
            except Exception as e:
                llm_response = f"[LLM call failed: {e}]"
                print(f"    LLM error: {e}")
        else:
            llm_response = "[No images available for analysis]"

        # 4. Save JSON with full attributes
        result_json = build_result_json(query, results, llm_response)
        json_path = OUTPUT_DIR / f"{slug}_results.json"
        json_path.write_text(
            json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"    Saved JSON → {json_path.name}")

        # 5. Save visualization
        vis_path = OUTPUT_DIR / f"{slug}_analysis.png"
        save_visualization(query, results, crops, llm_response, vis_path)
        print()

    print(f"Done. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all()
