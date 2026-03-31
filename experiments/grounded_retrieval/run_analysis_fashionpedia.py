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
import seaborn as sns
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fashionpedia.loaders import load_fashionpedia_catalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grounded_fashionpedia"

QUERY = "summer beach outfit"
TOP_K = 1000

MAX_CLUSTERS = 8
MIN_TOKEN_FREQ_RATIO = 0.03
RANDOM_STATE = 42

# Visualization settings
HEATMAP_TOP_N_ATTRS = 20
IMAGE_GRID_TOP_N = 12
IMAGE_GRID_COLS = 4
THUMBNAIL_SIZE = (128, 128)
LIFT_TOP_N = 25
LIFT_MIN_RETRIEVED_FREQ = 0.01


def make_output_dir(query_slug: str) -> Path:
    """Create and return outputs/grounded_fashionpedia/<query_slug>/."""
    out = BASE_OUTPUT_DIR / query_slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, 1e-12)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def safe_slug(text: str) -> str:
    return (
        text.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(",", "")
        .replace(".", "")
        .replace(":", "")
        .replace(";", "")
        .replace('"', "")
        .replace("'", "")
    )


def clean_value(value: object) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    if value_str.lower() in {"nan", "none", "null", "unknown"}:
        return None
    return value_str


def short_label(attr_name: str, attr_value: str, max_len: int = 32) -> str:
    label = f"{attr_name.replace('attr.', '')}={attr_value}"
    return label[:max_len] if len(label) > max_len else label


# ── RETRIEVAL ──────────────────────────────────────────────────────────────────

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

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        rows.append({
            "rank": rank,
            "item_id": item_ids[idx],
            "similarity_score": float(scores[idx]),
            "embedding_index": int(idx),
        })
    return pd.DataFrame(rows)


# ── METADATA TABLE ─────────────────────────────────────────────────────────────

def flatten_attribute_annotations(attr_annotations: Dict[str, Set[str]]) -> Dict[str, str]:
    """Flatten supercategory → {attr names} into columns like 'attr.textile pattern'."""
    flat: Dict[str, str] = {}
    for supercat, attr_set in attr_annotations.items():
        if not attr_set:
            continue
        flat[f"attr.{supercat}"] = "|".join(sorted(attr_set))
    return flat


def build_retrieved_metadata_df(retrieved_df: pd.DataFrame, catalog) -> pd.DataFrame:
    rows = []
    for _, row in retrieved_df.iterrows():
        item_id = row["item_id"]
        bbox = catalog.bboxes.get(item_id, [None, None, None, None])
        color_labels = catalog.color_annotations.get(item_id, [])

        flat_row: Dict = {
            "rank": int(row["rank"]),
            "item_id": item_id,
            "similarity_score": float(row["similarity_score"]),
            "embedding_index": int(row["embedding_index"]),
            "image_path": str(catalog.image_paths[item_id]) if item_id in catalog.image_paths else None,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "category": clean_value(catalog.category_annotations.get(item_id)),
            "color": "|".join(color_labels) if color_labels else None,
        }
        flat_row.update(flatten_attribute_annotations(catalog.attribute_annotations.get(item_id, {})))
        rows.append(flat_row)
    return pd.DataFrame(rows)


def infer_attribute_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "rank", "item_id", "similarity_score", "embedding_index",
        "image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    }
    return [col for col in df.columns if col not in excluded]


# ── ATTRIBUTE ANALYSIS ─────────────────────────────────────────────────────────

def split_multivalue(value: str) -> List[str]:
    candidates = [value]
    for sep in ["|", ",", ";"]:
        next_candidates = []
        for candidate in candidates:
            if sep in candidate:
                next_candidates.extend(candidate.split(sep))
            else:
                next_candidates.append(candidate)
        candidates = next_candidates
    return [c for item in candidates if (c := clean_value(item)) is not None]


def compute_attribute_frequencies(df: pd.DataFrame, attribute_columns: List[str]) -> pd.DataFrame:
    rows = []
    n_items = len(df)
    for col in attribute_columns:
        counts: Dict[str, int] = {}
        for raw_value in df[col]:
            value = clean_value(raw_value)
            if value is None:
                continue
            for part in split_multivalue(value):
                counts[part] = counts.get(part, 0) + 1
        for attr_value, count in counts.items():
            rows.append({
                "attribute_name": col,
                "attribute_value": attr_value,
                "count": count,
                "frequency": count / n_items if n_items > 0 else 0.0,
            })
    freq_df = pd.DataFrame(rows)
    if not freq_df.empty:
        freq_df = freq_df.sort_values(
            by=["attribute_name", "count"], ascending=[True, False]
        ).reset_index(drop=True)
    return freq_df


def shannon_entropy(probabilities: List[float]) -> float:
    probs = [p for p in probabilities if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


def compute_attribute_entropy(freq_df: pd.DataFrame) -> pd.DataFrame:
    if freq_df.empty:
        return pd.DataFrame(columns=[
            "attribute_name", "num_distinct_values", "entropy",
            "normalized_entropy", "top_value", "top_value_frequency",
        ])
    rows = []
    for attr_name, group in freq_df.groupby("attribute_name"):
        probs = group["frequency"].tolist()
        entropy = shannon_entropy(probs)
        num_values = len(group)
        max_entropy = math.log2(num_values) if num_values > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        top_row = group.sort_values("count", ascending=False).iloc[0]
        rows.append({
            "attribute_name": attr_name,
            "num_distinct_values": int(num_values),
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "top_value": top_row["attribute_value"],
            "top_value_frequency": float(top_row["frequency"]),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(by=["top_value_frequency", "normalized_entropy"], ascending=[False, True])
        .reset_index(drop=True)
    )


# ── CATALOG BASELINE FREQUENCIES (for lift) ────────────────────────────────────

def compute_catalog_attribute_frequencies(catalog) -> Dict[str, float]:
    """
    Frequency of each (column, value) pair across ALL items in the catalog.
    Returns dict keyed as "attr.textile pattern=floral" → fraction of items.
    """
    n_items = len(catalog.item_ids)
    if n_items == 0:
        return {}

    counts: Dict[str, int] = {}

    for item_id in catalog.item_ids:
        cat = catalog.category_annotations.get(item_id)
        if cat:
            key = f"category={cat}"
            counts[key] = counts.get(key, 0) + 1

        for color in catalog.color_annotations.get(item_id, []):
            key = f"color={color}"
            counts[key] = counts.get(key, 0) + 1

        for supercat, attr_set in catalog.attribute_annotations.get(item_id, {}).items():
            col = f"attr.{supercat}"
            for attr in attr_set:
                key = f"{col}={attr}"
                counts[key] = counts.get(key, 0) + 1

    return {k: v / n_items for k, v in counts.items()}


def compute_lift(freq_df: pd.DataFrame, catalog_freqs: Dict[str, float]) -> pd.DataFrame:
    """
    Lift = retrieved_frequency / catalog_frequency for each (attribute, value).
    Infinite lift (not present in catalog) is stored as float('inf').
    """
    rows = []
    for _, row in freq_df.iterrows():
        attr_name = row["attribute_name"]
        attr_value = row["attribute_value"]
        catalog_freq = catalog_freqs.get(f"{attr_name}={attr_value}", 0.0)
        retrieved_freq = float(row["frequency"])

        if catalog_freq < 1e-9:
            lift = float("inf") if retrieved_freq > 0 else 1.0
        else:
            lift = retrieved_freq / catalog_freq

        rows.append({
            "attribute_name": attr_name,
            "attribute_value": attr_value,
            "count": int(row["count"]),
            "retrieved_frequency": retrieved_freq,
            "catalog_frequency": catalog_freq,
            "lift": lift,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by="lift", ascending=False, key=lambda s: s.replace(float("inf"), 1e9)
        ).reset_index(drop=True)
    return df


# ── ATTRIBUTE-SPACE CLUSTERING ─────────────────────────────────────────────────

def build_attribute_documents(
    df: pd.DataFrame,
    attribute_columns: List[str],
    min_token_freq_ratio: float,
) -> Tuple[List[str], List[str]]:
    token_lists: List[List[str]] = []
    for _, row in df.iterrows():
        tokens: List[str] = []
        for col in attribute_columns:
            value = clean_value(row[col]) if col in row else None
            if value is None:
                continue
            for part in split_multivalue(value):
                tokens.append(f"{col}={part.lower().replace(' ', '_')}")
        token_lists.append(tokens)

    all_tokens = [t for tokens in token_lists for t in tokens]
    if not all_tokens:
        return ["" for _ in range(len(df))], []

    token_counts = pd.Series(all_tokens).value_counts()
    min_count = max(1, int(math.ceil(len(df) * min_token_freq_ratio)))
    kept_tokens = set(token_counts[token_counts >= min_count].index.tolist())

    docs = []
    for tokens in token_lists:
        filtered = [t for t in tokens if t in kept_tokens]
        docs.append(" ".join(filtered))
    return docs, sorted(kept_tokens)


def _run_kmeans_sweep(
    X,
    n_samples: int,
    max_clusters: int,
    random_state: int,
    metric: str = "euclidean",
) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
    best_k, best_score, best_labels = None, None, None
    upper_k = min(max_clusters, n_samples - 1)
    for k in range(2, upper_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(X, labels, metric=metric))
            if best_score is None or score > best_score:
                best_score, best_k, best_labels = score, k, labels
        except Exception as e:
            print(f"  KMeans k={k} failed: {e}")
    return best_k, best_score, best_labels


def cluster_by_attributes(
    df: pd.DataFrame,
    attribute_columns: List[str],
    max_clusters: int,
    min_token_freq_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int], Optional[float]]:
    docs, kept_tokens = build_attribute_documents(df, attribute_columns, min_token_freq_ratio)

    if not kept_tokens:
        clustered_df = df.copy()
        clustered_df["cluster_id"] = 0
        return clustered_df, pd.DataFrame([{"cluster_id": 0, "size": len(df)}]), None, None

    usable_mask = [bool(doc.strip()) for doc in docs]
    usable_df = df.loc[usable_mask].copy().reset_index(drop=True)
    usable_docs = [d for d in docs if d.strip()]

    if len(usable_docs) < 3:
        clustered_df = df.copy()
        clustered_df["cluster_id"] = 0
        return clustered_df, pd.DataFrame([{"cluster_id": 0, "size": len(df)}]), None, None

    X = CountVectorizer(binary=True).fit_transform(usable_docs)
    best_k, best_score, best_labels = _run_kmeans_sweep(
        X, len(usable_df), max_clusters, random_state, metric="euclidean"
    )

    clustered_df = df.copy()
    if best_labels is None:
        clustered_df["cluster_id"] = 0
        return clustered_df, pd.DataFrame([{"cluster_id": 0, "size": len(df)}]), None, None

    clustered_df["cluster_id"] = -1
    usable_df["cluster_id"] = best_labels
    clustered_df.loc[usable_mask, "cluster_id"] = usable_df["cluster_id"].values
    clustered_df.loc[~pd.Series(usable_mask).values, "cluster_id"] = -1

    summary_rows = [
        {"cluster_id": int(cid), "size": int(len(grp))}
        for cid, grp in usable_df.groupby("cluster_id")
    ]
    summary_df = (
        pd.DataFrame(summary_rows).sort_values("size", ascending=False).reset_index(drop=True)
    )
    return clustered_df, summary_df, best_k, best_score


def summarize_cluster_top_attributes(
    clustered_df: pd.DataFrame,
    attribute_columns: List[str],
    cluster_col: str = "cluster_id",
    top_n: int = 10,
) -> pd.DataFrame:
    rows = []
    valid_df = clustered_df[clustered_df[cluster_col] >= 0].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["cluster_id", "attribute_name", "attribute_value", "count", "frequency_in_cluster"]
        )
    for cluster_id, group in valid_df.groupby(cluster_col):
        cluster_size = len(group)
        counts: Dict[Tuple[str, str], int] = {}
        for col in attribute_columns:
            for raw_value in group[col]:
                value = clean_value(raw_value)
                if value is None:
                    continue
                for part in split_multivalue(value):
                    key = (col, part)
                    counts[key] = counts.get(key, 0) + 1
        for (attr_name, attr_value), count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            rows.append({
                "cluster_id": int(cluster_id),
                "attribute_name": attr_name,
                "attribute_value": attr_value,
                "count": int(count),
                "frequency_in_cluster": count / cluster_size if cluster_size > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def build_cluster_profile(
    clustered_df: pd.DataFrame,
    attribute_columns: List[str],
    cluster_col: str,
) -> List[Dict]:
    """
    For each cluster return a structured profile:
      - cluster_id, size
      - dominant_attributes: {attribute_column -> {top_value, frequency}}
    Used in both the JSON summary and console output.
    """
    valid = clustered_df[clustered_df[cluster_col] >= 0]
    profiles = []
    for cid in sorted(valid[cluster_col].unique()):
        group = valid[valid[cluster_col] == cid]
        cluster_size = len(group)
        dominant: Dict[str, Dict] = {}
        for col in attribute_columns:
            counts: Dict[str, int] = {}
            for raw in group[col]:
                v = clean_value(raw)
                if v is None:
                    continue
                for part in split_multivalue(v):
                    counts[part] = counts.get(part, 0) + 1
            if counts:
                top_val = max(counts, key=lambda k: counts[k])
                dominant[col] = {
                    "top_value": top_val,
                    "frequency": round(counts[top_val] / cluster_size, 4),
                }
        profiles.append({
            "cluster_id": int(cid),
            "size": cluster_size,
            "dominant_attributes": dominant,
        })
    return profiles


# ── EMBEDDING-SPACE CLUSTERING ─────────────────────────────────────────────────

def cluster_by_embeddings(
    metadata_df: pd.DataFrame,
    catalog,
    max_clusters: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int], Optional[float]]:
    """
    K-means clustering directly on L2-normalised FashionCLIP embeddings.
    Best k chosen by silhouette score (cosine metric, since vectors are unit-norm).
    Adds 'emb_cluster_id' column to metadata_df.
    """
    indices = metadata_df["embedding_index"].values
    embeddings = catalog.embeddings[indices].astype(np.float32)
    n = len(embeddings)

    if n < 3:
        clustered = metadata_df.copy()
        clustered["emb_cluster_id"] = 0
        return clustered, pd.DataFrame([{"emb_cluster_id": 0, "size": n}]), None, None

    print(f"  Sweeping k=2..{min(max_clusters, n - 1)} on {n} embeddings (cosine silhouette)...")
    best_k, best_score, best_labels = _run_kmeans_sweep(
        embeddings, n, max_clusters, random_state, metric="cosine"
    )

    clustered = metadata_df.copy()
    if best_labels is None:
        clustered["emb_cluster_id"] = 0
        return clustered, pd.DataFrame([{"emb_cluster_id": 0, "size": n}]), None, None

    clustered["emb_cluster_id"] = best_labels

    summary_rows = [
        {"emb_cluster_id": int(cid), "size": int(len(grp))}
        for cid, grp in clustered.groupby("emb_cluster_id")
    ]
    summary_df = (
        pd.DataFrame(summary_rows).sort_values("size", ascending=False).reset_index(drop=True)
    )
    return clustered, summary_df, best_k, best_score


# ── VISUALIZATIONS ─────────────────────────────────────────────────────────────

def _build_cluster_attr_matrix(
    df: pd.DataFrame,
    attribute_columns: List[str],
    cluster_col: str,
    top_n_attrs: int,
) -> Tuple[Optional[np.ndarray], List[str], List[int]]:
    """Build (n_clusters × n_attrs) frequency matrix for heatmap."""
    valid = df[df[cluster_col] >= 0]
    if valid.empty:
        return None, [], []

    # Global counts to pick the top attributes
    overall: Dict[Tuple[str, str], int] = {}
    for col in attribute_columns:
        for raw in valid[col]:
            v = clean_value(raw)
            if v is None:
                continue
            for part in split_multivalue(v):
                key = (col, part)
                overall[key] = overall.get(key, 0) + 1

    top_attrs = sorted(overall, key=lambda k: overall[k], reverse=True)[:top_n_attrs]
    if not top_attrs:
        return None, [], []

    cluster_ids = sorted(valid[cluster_col].unique())
    matrix = np.zeros((len(cluster_ids), len(top_attrs)), dtype=np.float32)

    for ci, cid in enumerate(cluster_ids):
        group = valid[valid[cluster_col] == cid]
        group_size = len(group)
        if group_size == 0:
            continue
        for ai, (col, part) in enumerate(top_attrs):
            cnt = 0
            for raw in group[col]:
                v = clean_value(raw)
                if v is None:
                    continue
                for p in split_multivalue(v):
                    if p == part:
                        cnt += 1
            matrix[ci, ai] = cnt / group_size

    col_labels = [short_label(col, part) for col, part in top_attrs]
    return matrix, col_labels, [int(c) for c in cluster_ids]


def plot_cluster_attribute_heatmap(
    clustered_df: pd.DataFrame,
    attribute_columns: List[str],
    cluster_col: str,
    output_path: Path,
    top_n_attrs: int = HEATMAP_TOP_N_ATTRS,
    title_suffix: str = "",
) -> None:
    """
    Heatmap: rows = clusters, columns = top attribute values,
    cell values = frequency of that attribute within the cluster.
    """
    matrix, col_labels, cluster_ids = _build_cluster_attr_matrix(
        clustered_df, attribute_columns, cluster_col, top_n_attrs
    )
    if matrix is None or len(cluster_ids) < 2:
        print(f"  Skipping heatmap ({cluster_col}): not enough clusters/data.")
        return

    row_labels = [f"C{cid}" for cid in cluster_ids]
    fig_w = max(14, len(col_labels) * 0.75)
    fig_h = max(4, len(cluster_ids) * 0.65)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Frequency in cluster", "shrink": 0.7},
    )
    title = f"Cluster × Attribute Heatmap{title_suffix}"
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Attribute value", fontsize=10)
    ax.set_ylabel("Cluster", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {output_path.name}")


def _load_crop(image_path: str, bbox: List, target_size: Tuple[int, int] = THUMBNAIL_SIZE) -> Optional[np.ndarray]:
    """Load full image, crop COCO bbox [x, y, w, h], resize, return HxWx3 array."""
    try:
        img = Image.open(image_path).convert("RGB")
        x, y, w, h = (max(0, int(v)) for v in bbox)
        w, h = max(1, w), max(1, h)
        crop = img.crop((x, y, x + w, y + h))
        crop = crop.resize(target_size, Image.LANCZOS)
        return np.array(crop)
    except Exception:
        return None


def plot_cluster_image_grids(
    clustered_df: pd.DataFrame,
    catalog,
    cluster_col: str,
    output_dir: Path,
    top_n: int = IMAGE_GRID_TOP_N,
    n_cols: int = IMAGE_GRID_COLS,
) -> None:
    """
    For each cluster, save a grid of the top-N images (by similarity score),
    cropped to their bounding boxes.
    """
    valid = clustered_df[clustered_df[cluster_col] >= 0].copy()
    if valid.empty:
        return

    for cid in sorted(valid[cluster_col].unique()):
        group = (
            valid[valid[cluster_col] == cid]
            .sort_values("similarity_score", ascending=False)
            .head(top_n)
        )

        crops, labels = [], []
        for _, row in group.iterrows():
            item_id = row["item_id"]
            img_path = row.get("image_path")
            bbox = catalog.bboxes.get(item_id, [0, 0, 64, 64])
            crop = _load_crop(img_path, bbox) if img_path else None
            crops.append(crop)
            category = clean_value(row.get("category")) or "?"
            labels.append(f"{category}\n{float(row['similarity_score']):.3f}")

        if not crops:
            continue

        n_rows = math.ceil(len(crops) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.8))
        axes = np.array(axes).reshape(-1)

        for i, (crop, lbl) in enumerate(zip(crops, labels)):
            ax = axes[i]
            if crop is not None:
                ax.imshow(crop)
            else:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#555555")
            ax.set_title(lbl, fontsize=7, pad=2)
            ax.axis("off")

        for i in range(len(crops), len(axes)):
            axes[i].axis("off")

        cluster_size = len(valid[valid[cluster_col] == cid])
        fig.suptitle(
            f"Cluster {int(cid)}  ({cluster_col})  —  top {len(crops)} of {cluster_size} items",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()

        out_path = output_dir / f"{cluster_col}_cluster{int(cid)}_images.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved image grid → {out_path.name}")


def plot_lift_bar_chart(
    lift_df: pd.DataFrame,
    output_path: Path,
    top_n: int = LIFT_TOP_N,
    min_retrieved_freq: float = LIFT_MIN_RETRIEVED_FREQ,
) -> None:
    """
    Horizontal bar chart of top attribute values by lift
    (retrieved_frequency / catalog_frequency).
    Attributes absent from the full catalog but present in the retrieval set
    are capped at a high finite value for display.
    """
    if lift_df.empty:
        return

    INF_PROXY = 50.0  # display cap for infinite-lift attributes

    finite = lift_df[
        (lift_df["lift"] != float("inf")) &
        (lift_df["retrieved_frequency"] >= min_retrieved_freq)
    ].copy()

    infinite = lift_df[
        (lift_df["lift"] == float("inf")) &
        (lift_df["retrieved_frequency"] >= min_retrieved_freq)
    ].copy()
    infinite["lift"] = INF_PROXY

    combined = (
        pd.concat([finite, infinite], ignore_index=True)
        .sort_values("lift", ascending=False)
        .head(top_n)
    )

    if combined.empty:
        return

    combined["label"] = combined.apply(
        lambda r: short_label(r["attribute_name"], r["attribute_value"], max_len=38), axis=1
    )
    combined["is_inf"] = combined["lift"] >= INF_PROXY

    # Reverse for bottom-to-top horizontal bars
    combined = combined.iloc[::-1].reset_index(drop=True)

    fig_h = max(6, len(combined) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    colors = ["#FF6B35" if inf else "#2196F3" for inf in combined["is_inf"]]
    ax.barh(combined["label"], combined["lift"], color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="baseline (lift = 1)")

    # Annotate infinite-lift bars
    for _, row in combined[combined["is_inf"]].iterrows():
        ax.text(
            INF_PROXY + 0.3, row.name,
            "∞ (not in catalog)", va="center", fontsize=7.5, color="#FF6B35"
        )

    ax.set_xlabel("Lift  (retrieved freq / catalog freq)", fontsize=10)
    ax.set_title("Enriched Attributes — Lift over Full Catalog", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="finite lift"),
        Patch(facecolor="#FF6B35", label=f"infinite lift (capped at {INF_PROXY})"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="baseline (lift = 1)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved lift chart → {output_path.name}")


# ── SAVE ───────────────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
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
    fashion_encoder = FashionCLIPEncoder()

    print("Loading Fashionpedia catalog...")
    catalog = load_fashionpedia_catalog()
    print(f"Catalog size: {len(catalog.item_ids)} items")

    print(f"Encoding query: '{query}'")
    query_embedding = fashion_encoder.encode_text(query)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    print(f"Retrieving top-{top_k} items...")
    retrieved_df = retrieve_top_k(
        query_embedding=query_embedding,
        item_ids=catalog.item_ids,
        item_embeddings=catalog.embeddings,
        top_k=top_k,
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    print("Building metadata table...")
    metadata_df = build_retrieved_metadata_df(retrieved_df, catalog)
    attribute_columns = infer_attribute_columns(metadata_df)
    print(f"Attribute columns ({len(attribute_columns)}): {attribute_columns}")

    # ── Attribute frequencies & entropy ───────────────────────────────────────
    print("Computing attribute frequencies...")
    freq_df = compute_attribute_frequencies(metadata_df, attribute_columns)

    print("Computing attribute entropy...")
    entropy_df = compute_attribute_entropy(freq_df)

    # ── Catalog baseline & lift ───────────────────────────────────────────────
    print("Computing full-catalog attribute frequencies (for lift)...")
    catalog_freqs = compute_catalog_attribute_frequencies(catalog)
    print(f"  Catalog unique (attr, value) pairs: {len(catalog_freqs)}")

    print("Computing attribute lift...")
    lift_df = compute_lift(freq_df, catalog_freqs)

    # ── Attribute-space clustering ─────────────────────────────────────────────
    print("Clustering by attribute tokens...")
    attr_clustered_df, attr_cluster_summary_df, attr_best_k, attr_silhouette = cluster_by_attributes(
        df=metadata_df,
        attribute_columns=attribute_columns,
        max_clusters=MAX_CLUSTERS,
        min_token_freq_ratio=MIN_TOKEN_FREQ_RATIO,
        random_state=RANDOM_STATE,
    )
    print(f"  Attribute clusters: best_k={attr_best_k}, silhouette={attr_silhouette}")

    attr_cluster_attr_df = summarize_cluster_top_attributes(
        attr_clustered_df, attribute_columns, cluster_col="cluster_id", top_n=10
    )

    # ── Embedding-space clustering ─────────────────────────────────────────────
    print("Clustering in embedding space...")
    emb_clustered_df, emb_cluster_summary_df, emb_best_k, emb_silhouette = cluster_by_embeddings(
        metadata_df=metadata_df,
        catalog=catalog,
        max_clusters=MAX_CLUSTERS,
        random_state=RANDOM_STATE,
    )
    print(f"  Embedding clusters: best_k={emb_best_k}, silhouette={emb_silhouette}")

    emb_cluster_attr_df = summarize_cluster_top_attributes(
        emb_clustered_df, attribute_columns, cluster_col="emb_cluster_id", top_n=10
    )

    # Merge both cluster columns into a single annotated frame
    full_clustered_df = attr_clustered_df.copy()
    full_clustered_df["emb_cluster_id"] = emb_clustered_df["emb_cluster_id"].values

    # ── Cluster profiles (dominant attribute per category, per cluster) ────────
    attr_profiles = build_cluster_profile(full_clustered_df, attribute_columns, "cluster_id")
    emb_profiles  = build_cluster_profile(full_clustered_df, attribute_columns, "emb_cluster_id")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\nSaving CSVs...")
    save_dataframe(retrieved_df,            output_dir / "retrieved.csv")
    save_dataframe(metadata_df,             output_dir / "retrieved_with_attributes.csv")
    save_dataframe(freq_df,                 output_dir / "attribute_frequencies.csv")
    save_dataframe(entropy_df,              output_dir / "attribute_entropy.csv")
    save_dataframe(lift_df,                 output_dir / "attribute_lift.csv")
    save_dataframe(full_clustered_df,       output_dir / "clustered_items.csv")
    save_dataframe(attr_cluster_summary_df, output_dir / "attr_cluster_summary.csv")
    save_dataframe(attr_cluster_attr_df,    output_dir / "attr_cluster_top_attributes.csv")
    save_dataframe(emb_cluster_summary_df,  output_dir / "emb_cluster_summary.csv")
    save_dataframe(emb_cluster_attr_df,     output_dir / "emb_cluster_top_attributes.csv")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")

    # 1. Heatmap — attribute clusters
    plot_cluster_attribute_heatmap(
        clustered_df=full_clustered_df,
        attribute_columns=attribute_columns,
        cluster_col="cluster_id",
        output_path=output_dir / "heatmap_attr_clusters.png",
        top_n_attrs=HEATMAP_TOP_N_ATTRS,
        title_suffix=" (attribute clustering)",
    )

    # 2. Heatmap — embedding clusters
    plot_cluster_attribute_heatmap(
        clustered_df=full_clustered_df,
        attribute_columns=attribute_columns,
        cluster_col="emb_cluster_id",
        output_path=output_dir / "heatmap_emb_clusters.png",
        top_n_attrs=HEATMAP_TOP_N_ATTRS,
        title_suffix=" (embedding clustering)",
    )

    # 3. Per-cluster image grids — embedding clusters
    print("  Generating per-cluster image grids (embedding clusters)...")
    plot_cluster_image_grids(
        clustered_df=full_clustered_df,
        catalog=catalog,
        cluster_col="emb_cluster_id",
        output_dir=output_dir,
        top_n=IMAGE_GRID_TOP_N,
        n_cols=IMAGE_GRID_COLS,
    )

    # 4. Lift bar chart
    plot_lift_bar_chart(
        lift_df=lift_df,
        output_path=output_dir / "attribute_lift_chart.png",
        top_n=LIFT_TOP_N,
        min_retrieved_freq=LIFT_MIN_RETRIEVED_FREQ,
    )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "dataset": "fashionpedia",
        "query": query,
        "top_k": top_k,
        "num_retrieved": int(len(retrieved_df)),
        "num_attribute_columns": int(len(attribute_columns)),
        "attribute_columns": attribute_columns,
        "attribute_clustering": {
            "best_k": attr_best_k,
            "silhouette_score": attr_silhouette,
            "clusters": attr_profiles,
        },
        "embedding_clustering": {
            "best_k": emb_best_k,
            "silhouette_score": emb_silhouette,
            "clusters": emb_profiles,
        },
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "summary.json")

    # ── Console summary ───────────────────────────────────────────────────────
    W = 80
    print("\n" + "=" * W)
    print("EXPERIMENT SUMMARY — FASHIONPEDIA")
    print("=" * W)
    print(f"Query            : {query}")
    print(f"Retrieved items  : {len(retrieved_df)}")
    print(f"Attribute columns: {len(attribute_columns)}")
    print(f"Attr  clusters   : best_k={attr_best_k}, silhouette={attr_silhouette}")
    print(f"Embed clusters   : best_k={emb_best_k}, silhouette={emb_silhouette}")

    print("\nTop dominant attributes across retrieved set (per category):")
    if entropy_df.empty:
        print("  None found.")
    else:
        for _, row in entropy_df.iterrows():
            print(
                f"  {row['attribute_name']:<42}  "
                f"top='{row['top_value']}' ({row['top_value_frequency']:.3f})  "
                f"entropy={row['normalized_entropy']:.3f}"
            )

    print("\nTop attributes by lift over full catalog:")
    if lift_df.empty:
        print("  None found.")
    else:
        shown = lift_df[lift_df["retrieved_frequency"] >= LIFT_MIN_RETRIEVED_FREQ].head(10)
        for _, row in shown.iterrows():
            lift_str = "∞" if row["lift"] == float("inf") else f"{row['lift']:.2f}x"
            print(
                f"  {row['attribute_name']}={row['attribute_value']}: "
                f"lift={lift_str}  retrieved={row['retrieved_frequency']:.3f}"
            )

    def _print_cluster_profiles(profiles: List[Dict], label: str) -> None:
        print(f"\n{label}:")
        for p in profiles:
            print(f"  Cluster {p['cluster_id']}  (size={p['size']})")
            for col, info in p["dominant_attributes"].items():
                col_short = col.replace("attr.", "")
                print(f"    {col_short:<38}  {info['top_value']}  ({info['frequency']:.3f})")

    _print_cluster_profiles(attr_profiles, "Attribute clusters — dominant value per category")
    _print_cluster_profiles(emb_profiles,  "Embedding clusters — dominant value per category")

    print(f"\nAll outputs saved to: {output_dir}")


run_experiment(query=QUERY, top_k=TOP_K)
