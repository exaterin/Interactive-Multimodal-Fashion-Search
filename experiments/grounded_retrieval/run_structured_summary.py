from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fashionpedia.loaders import load_fashionpedia_catalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder

# ── CONFIG ─────────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "structured_fashionpedia"

QUERY = "denim coat"
TOP_K = 1000

# How many top categories to include in per-group analysis and visualisation
TOP_CATEGORIES_N = 8
# Minimum fraction of retrieved set for a category to get a full group profile
MIN_CATEGORY_FREQ = 0.01
# Top attribute values to show per group profile panel
TOP_ATTRS_PER_GROUP = 8
# Attributes shown in the global lift chart
LIFT_CHART_TOP_N = 25
LIFT_MIN_RETRIEVED_FREQ = 0.01
# Example item grids
EXAMPLE_ITEMS_PER_GROUP = 6
THUMBNAIL_SIZE = (128, 128)


def make_output_dir(query_slug: str) -> Path:
    out = BASE_OUTPUT_DIR / query_slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_slug(text: str) -> str:
    return (
        text.lower().strip()
        .replace(" ", "_").replace("/", "_").replace("\\", "_")
        .replace(",", "").replace(".", "").replace(":", "")
        .replace(";", "").replace('"', "").replace("'", "")
    )


def clean_value(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "unknown"}:
        return None
    return s


def split_multivalue(value: str) -> List[str]:
    parts = [value]
    for sep in ["|", ",", ";"]:
        parts = [p for raw in parts for p in (raw.split(sep) if sep in raw else [raw])]
    return [c for p in parts if (c := clean_value(p)) is not None]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, 1e-12)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def retrieve_top_k(
    query_embedding: np.ndarray,
    item_ids: List[str],
    item_embeddings: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    query_embedding = l2_normalize(query_embedding.astype(np.float32))
    item_embeddings = l2_normalize(item_embeddings.astype(np.float32))
    scores = item_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(-scores)[:top_k]
    return pd.DataFrame([
        {
            "rank": rank,
            "item_id": item_ids[idx],
            "similarity_score": float(scores[idx]),
            "embedding_index": int(idx),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ])


def build_retrieved_metadata_df(retrieved_df: pd.DataFrame, catalog) -> pd.DataFrame:
    rows = []
    for _, row in retrieved_df.iterrows():
        item_id = row["item_id"]
        bbox = catalog.bboxes.get(item_id, [None, None, None, None])
        color_labels = catalog.color_annotations.get(item_id, [])
        flat: Dict = {
            "rank": int(row["rank"]),
            "item_id": item_id,
            "similarity_score": float(row["similarity_score"]),
            "embedding_index": int(row["embedding_index"]),
            "image_path": str(catalog.image_paths[item_id]) if item_id in catalog.image_paths else None,
            "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3],
            "category": clean_value(catalog.category_annotations.get(item_id)),
            "color": "|".join(color_labels) if color_labels else None,
        }
        for supercat, attr_set in catalog.attribute_annotations.get(item_id, {}).items():
            if attr_set:
                flat[f"attr.{supercat}"] = "|".join(sorted(attr_set))
        rows.append(flat)
    return pd.DataFrame(rows)


def infer_attribute_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "rank", "item_id", "similarity_score", "embedding_index",
        "image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    }
    return [c for c in df.columns if c not in excluded]


def compute_attribute_frequencies(df: pd.DataFrame, attribute_columns: List[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in attribute_columns:
        counts: Dict[str, int] = {}
        for raw in df[col]:
            v = clean_value(raw)
            if v is None:
                continue
            for part in split_multivalue(v):
                counts[part] = counts.get(part, 0) + 1
        for val, cnt in counts.items():
            rows.append({"attribute_name": col, "attribute_value": val,
                         "count": cnt, "frequency": cnt / n if n else 0.0})
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["attribute_name", "count"], ascending=[True, False]).reset_index(drop=True)
    return df_out


def compute_catalog_attribute_frequencies(catalog) -> Dict[str, float]:
    """fraction of catalog items that have (col, value), keyed as 'col=value'."""
    n = len(catalog.item_ids)
    if n == 0:
        return {}
    counts: Dict[str, int] = {}
    for item_id in catalog.item_ids:
        cat = catalog.category_annotations.get(item_id)
        if cat:
            counts[f"category={cat}"] = counts.get(f"category={cat}", 0) + 1
        for c in catalog.color_annotations.get(item_id, []):
            k = f"color={c}"
            counts[k] = counts.get(k, 0) + 1
        for supercat, attr_set in catalog.attribute_annotations.get(item_id, {}).items():
            col = f"attr.{supercat}"
            for a in attr_set:
                k = f"{col}={a}"
                counts[k] = counts.get(k, 0) + 1
    return {k: v / n for k, v in counts.items()}


def _lift(retrieved_freq: float, catalog_freq: float) -> float:
    if catalog_freq < 1e-9:
        return float("inf") if retrieved_freq > 0 else 1.0
    return retrieved_freq / catalog_freq


def compute_lift_df(freq_df: pd.DataFrame, catalog_freqs: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, row in freq_df.iterrows():
        col, val = row["attribute_name"], row["attribute_value"]
        r_freq = float(row["frequency"])
        c_freq = catalog_freqs.get(f"{col}={val}", 0.0)
        rows.append({
            "attribute_name": col,
            "attribute_value": val,
            "count": int(row["count"]),
            "retrieved_frequency": r_freq,
            "catalog_frequency": c_freq,
            "lift": _lift(r_freq, c_freq),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "lift", ascending=False,
            key=lambda s: s.replace(float("inf"), 1e9)
        ).reset_index(drop=True)
    return df



def compute_category_summary(
    metadata_df: pd.DataFrame,
    catalog_freqs: Dict[str, float],
) -> pd.DataFrame:
    """
    One row per category: count in retrieved set, retrieved_frequency,
    catalog_frequency, lift, and rank.
    """
    n = len(metadata_df)
    counts: Dict[str, int] = {}
    for raw in metadata_df["category"]:
        v = clean_value(raw)
        if v:
            counts[v] = counts.get(v, 0) + 1

    rows = []
    for cat, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        r_freq = cnt / n if n else 0.0
        c_freq = catalog_freqs.get(f"category={cat}", 0.0)
        rows.append({
            "category": cat,
            "count": cnt,
            "retrieved_frequency": r_freq,
            "catalog_frequency": c_freq,
            "lift": _lift(r_freq, c_freq),
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df



def _dominant_attributes_for_group(
    group_df: pd.DataFrame,
    attribute_columns: List[str],
    catalog_freqs: Dict[str, float],
    top_n: int,
) -> List[Dict]:
    """
    For a subset of metadata_df (one category group) return the top-N
    (attr, value) pairs ranked by frequency within the group, annotated with lift.
    Only non-category attribute columns are considered.
    """
    n = len(group_df)
    attr_cols = [c for c in attribute_columns if c != "category"]
    counts: Dict[Tuple[str, str], int] = {}
    for col in attr_cols:
        for raw in group_df[col]:
            v = clean_value(raw)
            if v is None:
                continue
            for part in split_multivalue(v):
                counts[(col, part)] = counts.get((col, part), 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []
    for (col, val), cnt in ranked:
        r_freq = cnt / n if n else 0.0
        c_freq = catalog_freqs.get(f"{col}={val}", 0.0)
        result.append({
            "attribute_name": col,
            "attribute_value": val,
            "count": cnt,
            "frequency_in_group": round(r_freq, 4),
            "catalog_frequency": round(c_freq, 4),
            "lift": round(_lift(r_freq, c_freq), 4),
        })
    return result


def build_interpretation(
    category: str,
    size: int,
    total_retrieved: int,
    category_lift: float,
    dominant_attrs: List[Dict],
) -> str:
    """
    Generate a short natural-language description of a category group.
    """
    pct = 100 * size / total_retrieved if total_retrieved else 0.0

    # Category lift phrase
    if category_lift == float("inf"):
        lift_phrase = "not present in catalog baseline"
    elif category_lift >= 5:
        lift_phrase = f"strongly over-represented ({category_lift:.1f}× catalog)"
    elif category_lift >= 2:
        lift_phrase = f"over-represented ({category_lift:.1f}× catalog)"
    elif category_lift >= 0.8:
        lift_phrase = f"proportional to catalog ({category_lift:.1f}×)"
    else:
        lift_phrase = f"under-represented ({category_lift:.1f}× catalog)"

    # Top attribute phrases — max 3, prioritise high lift
    high_lift = sorted(dominant_attrs, key=lambda d: d["lift"] if d["lift"] != float("inf") else 1e9, reverse=True)
    attr_phrases = []
    for d in high_lift[:3]:
        col_clean = d["attribute_name"].replace("attr.", "")
        lift_val = d["lift"]
        lift_tag = "" if lift_val == float("inf") or lift_val < 2 else f" [{lift_val:.1f}× lift]"
        attr_phrases.append(f"{d['attribute_value']} {col_clean} ({d['frequency_in_group']:.0%}){lift_tag}")

    attr_str = ", ".join(attr_phrases) if attr_phrases else "no dominant attributes"

    return (
        f"{size} {category} items ({pct:.1f}% of retrieved), {lift_phrase}. "
        f"Predominantly: {attr_str}."
    )


def compute_group_attribute_profiles(
    metadata_df: pd.DataFrame,
    attribute_columns: List[str],
    category_summary_df: pd.DataFrame,
    catalog_freqs: Dict[str, float],
    top_categories_n: int,
    min_category_freq: float,
    top_attrs_per_group: int,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    For each qualifying category build:
      - A flat DataFrame of per-group attribute rows (for CSV export)
      - A list of rich group-profile dicts (for JSON summary and console output)
    """
    total = len(metadata_df)
    qualifying = category_summary_df[
        category_summary_df["retrieved_frequency"] >= min_category_freq
    ].head(top_categories_n)

    flat_rows = []
    profiles = []

    for _, cat_row in qualifying.iterrows():
        category = cat_row["category"]
        size = int(cat_row["count"])
        cat_lift = float(cat_row["lift"])

        group_df = metadata_df[metadata_df["category"] == category]
        dom_attrs = _dominant_attributes_for_group(
            group_df, attribute_columns, catalog_freqs, top_attrs_per_group
        )
        interpretation = build_interpretation(category, size, total, cat_lift, dom_attrs)

        profiles.append({
            "category": category,
            "group_size": size,
            "retrieved_frequency": round(float(cat_row["retrieved_frequency"]), 4),
            "catalog_frequency": round(float(cat_row["catalog_frequency"]), 4),
            "category_lift": round(cat_lift, 4) if cat_lift != float("inf") else "inf",
            "dominant_attributes": dom_attrs,
            "interpretation": interpretation,
        })

        for d in dom_attrs:
            flat_rows.append({"category": category, **d})

    return pd.DataFrame(flat_rows), profiles


# Vizualizations

def plot_category_distribution(
    category_summary_df: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_CATEGORIES_N,
) -> None:
    """
    Dual-axis horizontal bar chart:
      left panel  – retrieved frequency (bars)
      right panel – category lift over catalog (bars coloured by magnitude)
    """
    df = category_summary_df.head(top_n).copy().iloc[::-1].reset_index(drop=True)
    if df.empty:
        return

    fig, (ax_freq, ax_lift) = plt.subplots(1, 2, figsize=(13, max(4, len(df) * 0.55)))
    fig.suptitle("Category Distribution in Retrieved Set", fontsize=13, y=1.01)

    # Left: retrieved frequency
    ax_freq.barh(df["category"], df["retrieved_frequency"], color="#2196F3",
                 edgecolor="white", linewidth=0.5)
    if "catalog_frequency" in df.columns:
        ax_freq.barh(df["category"], df["catalog_frequency"], color="#BBDEFB",
                     edgecolor="white", linewidth=0.5, label="catalog freq")
        ax_freq.legend(fontsize=8)
    ax_freq.set_xlabel("Frequency", fontsize=9)
    ax_freq.set_title("Retrieved freq  (blue) vs catalog (light)", fontsize=9)
    ax_freq.spines["top"].set_visible(False)
    ax_freq.spines["right"].set_visible(False)

    # Right: lift
    INF_PROXY = df[df["lift"] != float("inf")]["lift"].max() * 1.15 if not df[df["lift"] != float("inf")].empty else 10.0
    lifts = df["lift"].replace(float("inf"), INF_PROXY)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=max(lifts.max(), 2.0))
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in lifts]
    bars = ax_lift.barh(df["category"], lifts, color=colors, edgecolor="white", linewidth=0.5)
    ax_lift.axvline(1.0, color="black", linestyle="--", linewidth=1, label="baseline")
    ax_lift.set_xlabel("Lift (retrieved / catalog freq)", fontsize=9)
    ax_lift.set_title("Category lift over full catalog", fontsize=9)
    ax_lift.legend(fontsize=8)
    ax_lift.spines["top"].set_visible(False)
    ax_lift.spines["right"].set_visible(False)

    # Mark inf-lift bars
    for i, (lift_raw, bar) in enumerate(zip(df["lift"], bars)):
        if lift_raw == float("inf"):
            ax_lift.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                         "∞", va="center", fontsize=9, color="darkgreen")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved category distribution → {output_path.name}")


def plot_group_attribute_profiles(
    group_profiles: List[Dict],
    output_path: Path,
    top_attrs: int = TOP_ATTRS_PER_GROUP,
) -> None:
    """
    Grid of subplots — one panel per category group.
    Each panel shows top attributes as horizontal bars coloured by lift:
      green  = lift > 2   (enriched)
      grey   = lift ≈ 1   (neutral)
      salmon = lift < 0.5 (depleted)
    """
    profiles = [p for p in group_profiles if p["dominant_attributes"]]
    if not profiles:
        return

    n_panels = len(profiles)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 3.8))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Per-Category Dominant Attributes  (bar = freq in group, colour = lift)",
                 fontsize=12, y=1.01)

    def _lift_color(lift: float) -> str:
        if lift == float("inf") or lift >= 3:
            return "#2E7D32"   # deep green
        if lift >= 1.5:
            return "#81C784"   # light green
        if lift >= 0.7:
            return "#90A4AE"   # grey
        return "#EF9A9A"       # salmon

    for ax, profile in zip(axes, profiles):
        attrs = profile["dominant_attributes"][:top_attrs]
        if not attrs:
            ax.axis("off")
            continue

        labels = []
        freqs = []
        colors = []
        for a in reversed(attrs):          # reversed so highest is at top
            col_short = a["attribute_name"].replace("attr.", "")
            labels.append(f"{a['attribute_value']}\n({col_short})")
            freqs.append(a["frequency_in_group"])
            colors.append(_lift_color(a["lift"]))

        ax.barh(labels, freqs, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Freq in group", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        cat_lift = profile["category_lift"]
        lift_str = "∞" if cat_lift == "inf" else f"{cat_lift:.1f}×"
        ax.set_title(
            f"{profile['category']}\n"
            f"n={profile['group_size']}  cat-lift={lift_str}",
            fontsize=8.5, pad=4,
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    # Legend for lift colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", label="lift ≥ 3×"),
        Patch(facecolor="#81C784", label="lift 1.5–3×"),
        Patch(facecolor="#90A4AE", label="lift ≈ 1×"),
        Patch(facecolor="#EF9A9A", label="lift < 0.7×"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group attribute profiles → {output_path.name}")


def plot_global_lift_chart(
    lift_df: pd.DataFrame,
    output_path: Path,
    top_n: int = LIFT_CHART_TOP_N,
    min_retrieved_freq: float = LIFT_MIN_RETRIEVED_FREQ,
) -> None:
    """
    Horizontal bar chart of top attribute values by global lift.
    """
    if lift_df.empty:
        return

    INF_PROXY = 50.0

    finite = lift_df[
        (lift_df["lift"] != float("inf")) &
        (lift_df["retrieved_frequency"] >= min_retrieved_freq)
    ].copy()
    infinite = lift_df[
        (lift_df["lift"] == float("inf")) &
        (lift_df["retrieved_frequency"] >= min_retrieved_freq)
    ].copy()
    infinite = infinite.copy()
    infinite["lift"] = INF_PROXY

    combined = (
        pd.concat([finite, infinite], ignore_index=True)
        .sort_values("lift", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    if combined.empty:
        return

    combined["label"] = combined.apply(
        lambda r: f"{r['attribute_name'].replace('attr.', '')}={r['attribute_value']}"[:40],
        axis=1,
    )
    combined["is_inf"] = combined["lift"] >= INF_PROXY

    fig_h = max(6, len(combined) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    colors = ["#FF6B35" if inf else "#2196F3" for inf in combined["is_inf"]]
    ax.barh(combined["label"], combined["lift"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)

    for _, row in combined[combined["is_inf"]].iterrows():
        ax.text(INF_PROXY + 0.5, row.name, "∞ (not in catalog)",
                va="center", fontsize=7.5, color="#FF6B35")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2196F3", label="finite lift"),
        Patch(facecolor="#FF6B35", label=f"∞ lift (capped at {INF_PROXY})"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="baseline (lift=1)"),
    ], fontsize=9, loc="lower right")

    ax.set_xlabel("Lift  (retrieved freq / catalog freq)", fontsize=10)
    ax.set_title("Enriched Attributes — Global Lift over Full Catalog", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved global lift chart → {output_path.name}")


def _load_crop(
    image_path: str,
    bbox: List,
    target_size: Tuple[int, int] = THUMBNAIL_SIZE,
) -> Optional[np.ndarray]:
    """Load full image, crop COCO bbox [x, y, w, h], resize to target_size."""
    try:
        img = Image.open(image_path).convert("RGB")
        x, y, w, h = (max(0, int(v)) for v in bbox)
        w, h = max(1, w), max(1, h)
        crop = img.crop((x, y, x + w, y + h))
        return np.array(crop.resize(target_size, Image.LANCZOS))
    except Exception:
        return None


def plot_group_example_items(
    metadata_df: pd.DataFrame,
    catalog,
    group_profiles: List[Dict],
    output_path: Path,
    top_n: int = EXAMPLE_ITEMS_PER_GROUP,
) -> None:
    """
    One row per category group. Each row has:
      - a text column on the left  : category, group size, lift, top 3 attributes
      - top_n image columns        : highest-scoring bbox crops for that category,
                                     labelled with their similarity score

    Gives an at-a-glance visual sense of what each group actually looks like.
    """
    if not group_profiles:
        return

    n_groups = len(group_profiles)
    cell_w, cell_h = 2.2, 2.8
    text_w_ratio = 1.4          # text column is 1.4× the width of one image cell

    fig = plt.figure(figsize=(text_w_ratio * cell_w + top_n * cell_w, n_groups * cell_h))
    gs = fig.add_gridspec(
        n_groups, top_n + 1,
        width_ratios=[text_w_ratio] + [1] * top_n,
        hspace=0.08,
        wspace=0.04,
    )
    fig.suptitle("Example Items per Category Group  (sorted by similarity score)",
                 fontsize=11, y=1.005)

    for row_idx, profile in enumerate(group_profiles):
        category = profile["category"]
        size = profile["group_size"]
        cat_lift = profile["category_lift"]
        lift_str = "∞" if cat_lift == "inf" else f"{cat_lift:.1f}×"

        # Left text cell — group summary
        attr_lines = []
        for a in profile["dominant_attributes"][:3]:
            col_short = a["attribute_name"].replace("attr.", "")[:14]
            attr_lines.append(
                f"• {a['attribute_value']} ({col_short}): {a['frequency_in_group']:.0%}"
            )
        info = f"{category}\nn={size}   lift={lift_str}\n\n" + "\n".join(attr_lines)

        ax_text = fig.add_subplot(gs[row_idx, 0])
        ax_text.axis("off")
        ax_text.text(
            0.5, 0.5, info,
            ha="center", va="center",
            fontsize=7.5, linespacing=1.4,
            transform=ax_text.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#F5F5F5",
                edgecolor="#BDBDBD",
                linewidth=0.8,
            ),
        )

        # Image cells — top-N items by similarity score
        group_rows = (
            metadata_df[metadata_df["category"] == category]
            .sort_values("similarity_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        for col_idx in range(top_n):
            ax_img = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax_img.axis("off")

            if col_idx >= len(group_rows):
                ax_img.set_facecolor("#EEEEEE")
                continue

            item_row = group_rows.iloc[col_idx]
            item_id = item_row["item_id"]
            img_path = item_row.get("image_path")
            bbox = catalog.bboxes.get(item_id, [0, 0, 64, 64])
            crop = _load_crop(img_path, bbox) if img_path else None

            if crop is not None:
                ax_img.imshow(crop)
                ax_img.set_title(
                    f"{float(item_row['similarity_score']):.3f}",
                    fontsize=6.5, pad=2,
                )
            else:
                ax_img.set_facecolor("#E0E0E0")
                ax_img.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax_img.transAxes, fontsize=7, color="#757575")

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved example items → {output_path.name}")


# ── SAVE ───────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def run_experiment(query: str, top_k: int) -> None:
    query_slug = safe_slug(query)
    output_dir = make_output_dir(query_slug)
    print(f"Output directory: {output_dir}")

    # ── Load ──────────────────────────────────────────────────────────────────
    encoder = FashionCLIPEncoder()
    print("Loading Fashionpedia catalog...")
    catalog = load_fashionpedia_catalog()
    print(f"Catalog: {len(catalog.item_ids)} items")

    # ── Retrieve ──────────────────────────────────────────────────────────────
    print(f"Encoding query: '{query}'")
    query_emb = encoder.encode_text(query)

    print(f"Retrieving top-{top_k} items...")
    retrieved_df = retrieve_top_k(query_emb, catalog.item_ids, catalog.embeddings, top_k)

    # ── Metadata ──────────────────────────────────────────────────────────────
    print("Building metadata table...")
    metadata_df = build_retrieved_metadata_df(retrieved_df, catalog)
    attribute_columns = infer_attribute_columns(metadata_df)

    # ── Frequencies ───────────────────────────────────────────────────────────
    print("Computing attribute frequencies...")
    freq_df = compute_attribute_frequencies(metadata_df, attribute_columns)

    # ── Catalog baseline & lift ───────────────────────────────────────────────
    print("Computing catalog baseline frequencies...")
    catalog_freqs = compute_catalog_attribute_frequencies(catalog)
    print(f"  {len(catalog_freqs)} unique (attr, value) pairs in catalog")

    print("Computing attribute lift...")
    lift_df = compute_lift_df(freq_df, catalog_freqs)

    # ── Category summary ──────────────────────────────────────────────────────
    print("Computing category summary...")
    category_summary_df = compute_category_summary(metadata_df, catalog_freqs)

    # ── Per-group attribute profiles ──────────────────────────────────────────
    print("Building per-group attribute profiles...")
    group_attr_df, group_profiles = compute_group_attribute_profiles(
        metadata_df=metadata_df,
        attribute_columns=attribute_columns,
        category_summary_df=category_summary_df,
        catalog_freqs=catalog_freqs,
        top_categories_n=TOP_CATEGORIES_N,
        min_category_freq=MIN_CATEGORY_FREQ,
        top_attrs_per_group=TOP_ATTRS_PER_GROUP,
    )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\nSaving CSVs...")
    save_csv(retrieved_df,        output_dir / "retrieved.csv")
    save_csv(metadata_df,         output_dir / "retrieved_with_attributes.csv")
    save_csv(freq_df,             output_dir / "attribute_frequencies.csv")
    save_csv(lift_df,             output_dir / "attribute_lift.csv")
    save_csv(category_summary_df, output_dir / "category_summary.csv")
    save_csv(group_attr_df,       output_dir / "group_attribute_profiles.csv")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")

    plot_category_distribution(
        category_summary_df,
        output_dir / "category_distribution.png",
        top_n=TOP_CATEGORIES_N,
    )

    plot_group_attribute_profiles(
        group_profiles,
        output_dir / "group_attribute_profiles.png",
        top_attrs=TOP_ATTRS_PER_GROUP,
    )

    plot_global_lift_chart(
        lift_df,
        output_dir / "global_lift_chart.png",
        top_n=LIFT_CHART_TOP_N,
        min_retrieved_freq=LIFT_MIN_RETRIEVED_FREQ,
    )

    plot_group_example_items(
        metadata_df=metadata_df,
        catalog=catalog,
        group_profiles=group_profiles,
        output_path=output_dir / "group_example_items.png",
        top_n=EXAMPLE_ITEMS_PER_GROUP,
    )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    n_ret = len(retrieved_df)
    top_lift_attrs = [
        {
            "attribute_name": r["attribute_name"],
            "attribute_value": r["attribute_value"],
            "retrieved_frequency": r["retrieved_frequency"],
            "catalog_frequency": r["catalog_frequency"],
            "lift": r["lift"] if r["lift"] != float("inf") else "inf",
        }
        for _, r in lift_df[lift_df["retrieved_frequency"] >= LIFT_MIN_RETRIEVED_FREQ]
        .head(15).iterrows()
    ]

    summary = {
        "dataset": "fashionpedia",
        "query": query,
        "top_k": top_k,
        "num_retrieved": n_ret,
        "num_attribute_columns": len(attribute_columns),
        "attribute_columns": attribute_columns,
        "dominant_categories": [
            {
                "rank": int(r["rank"]),
                "category": r["category"],
                "count": int(r["count"]),
                "retrieved_frequency": round(float(r["retrieved_frequency"]), 4),
                "catalog_frequency": round(float(r["catalog_frequency"]), 4),
                "lift": round(float(r["lift"]), 4) if r["lift"] != float("inf") else "inf",
            }
            for _, r in category_summary_df.head(TOP_CATEGORIES_N).iterrows()
        ],
        "dominant_attributes_global": top_lift_attrs,
        "category_groups": group_profiles,
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "summary.json")

    # ── Console output ────────────────────────────────────────────────────────
    W = 80
    print("\n" + "=" * W)
    print("STRUCTURED SUMMARY — FASHIONPEDIA")
    print("=" * W)
    print(f"Query           : {query}")
    print(f"Retrieved items : {n_ret}")
    print(f"Distinct cats   : {len(category_summary_df)}")

    print(f"\n{'─'*W}")
    print("DOMINANT CATEGORIES")
    print(f"{'─'*W}")
    print(f"  {'#':<4} {'Category':<35} {'Count':>6}  {'Ret%':>6}  {'Cat%':>6}  {'Lift':>7}")
    print(f"  {'─'*4} {'─'*35} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")
    for _, r in category_summary_df.head(TOP_CATEGORIES_N).iterrows():
        lift_str = "∞" if r["lift"] == float("inf") else f"{r['lift']:.2f}×"
        print(
            f"  {int(r['rank']):<4} {r['category']:<35} "
            f"{int(r['count']):>6}  "
            f"{r['retrieved_frequency']:>5.1%}  "
            f"{r['catalog_frequency']:>5.1%}  "
            f"{lift_str:>7}"
        )

    print(f"\n{'─'*W}")
    print("TOP ATTRIBUTES BY LIFT (global, retrieved set)")
    print(f"{'─'*W}")
    shown = lift_df[lift_df["retrieved_frequency"] >= LIFT_MIN_RETRIEVED_FREQ].head(12)
    for _, r in shown.iterrows():
        lift_str = "∞" if r["lift"] == float("inf") else f"{r['lift']:.2f}×"
        print(
            f"  {r['attribute_name'].replace('attr.',''):<38} = {r['attribute_value']:<25} "
            f"lift={lift_str:>7}  ret={r['retrieved_frequency']:.3f}"
        )

    print(f"\n{'─'*W}")
    print("CATEGORY GROUPS")
    print(f"{'─'*W}")
    for p in group_profiles:
        cat_lift_str = "∞" if p["category_lift"] == "inf" else f"{p['category_lift']:.2f}×"
        print(f"\n  ── {p['category']}  "
              f"(n={p['group_size']}, ret={p['retrieved_frequency']:.1%}, lift={cat_lift_str})")
        print(f"     {p['interpretation']}")
        print(f"     {'Attribute':<42} {'Value':<25} {'Grp%':>6}  {'Lift':>7}")
        for a in p["dominant_attributes"]:
            lift_str = "∞" if a["lift"] == float("inf") else f"{a['lift']:.2f}×"
            col_short = a["attribute_name"].replace("attr.", "")
            print(
                f"     {col_short:<42} {a['attribute_value']:<25} "
                f"{a['frequency_in_group']:>5.1%}  {lift_str:>7}"
            )

    print(f"\nAll outputs saved to: {output_dir}")


run_experiment(query=QUERY, top_k=TOP_K)
