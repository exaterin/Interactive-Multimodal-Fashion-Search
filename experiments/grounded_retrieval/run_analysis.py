# grounded/grounded_retrieval_experiment.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deepfashion.loaders import load_catalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder




OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grounded"

QUERY = "black sunglasses"
TOP_K = 1000

MAX_CLUSTERS = 8
MIN_TOKEN_FREQ_RATIO = 0.03
RANDOM_STATE = 42


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    lowered = value_str.lower()
    if lowered in {"nan", "none", "null", "unknown"}:
        return None

    return value_str


def normalize_annotation_dict(annotation_dict: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Makes sure annotation values are strings and removes empty ones.
    """
    if not annotation_dict:
        return {}

    result: Dict[str, str] = {}
    for key, value in annotation_dict.items():
        clean = clean_value(value)
        if clean is not None:
            result[str(key)] = clean

    return result


def flatten_annotation_fields(
    prefix: str,
    annotation_dict: Dict[str, str],
) -> Dict[str, str]:
    """
    Converts nested annotation dict into flat columns like:
    fabric.texture=lace
    shape.length=midi
    """
    flat: Dict[str, str] = {}

    for key, value in annotation_dict.items():
        column_name = f"{prefix}.{key}"
        flat[column_name] = value

    return flat


# RETRIEVAL

def retrieve_top_k(
    query_embedding: np.ndarray,
    image_ids: List[str],
    image_embeddings: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    query_embedding = l2_normalize(query_embedding.astype(np.float32))
    image_embeddings = l2_normalize(image_embeddings.astype(np.float32))

    scores = image_embeddings @ query_embedding
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(-scores)[:top_k]

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        rows.append(
            {
                "rank": rank,
                "image_id": image_ids[idx],
                "similarity_score": float(scores[idx]),
                "embedding_index": int(idx),
            }
        )

    return pd.DataFrame(rows)


# BUILD METADATA TABLE

def build_retrieved_metadata_df(
    retrieved_df: pd.DataFrame,
    catalog,
) -> pd.DataFrame:
    rows = []

    for _, row in retrieved_df.iterrows():
        image_id = row["image_id"]

        fabric_dict = normalize_annotation_dict(catalog.fabric_annotations.get(image_id, {}))
        pattern_dict = normalize_annotation_dict(catalog.pattern_annotations.get(image_id, {}))
        shape_dict = normalize_annotation_dict(catalog.shape_annotations.get(image_id, {}))

        flat_row = {
            "rank": int(row["rank"]),
            "image_id": image_id,
            "similarity_score": float(row["similarity_score"]),
            "image_path": str(catalog.image_paths[image_id]) if image_id in catalog.image_paths else None,
        }

        flat_row.update(flatten_annotation_fields("fabric", fabric_dict))
        flat_row.update(flatten_annotation_fields("pattern", pattern_dict))
        flat_row.update(flatten_annotation_fields("shape", shape_dict))

        rows.append(flat_row)

    return pd.DataFrame(rows)


def infer_attribute_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"rank", "image_id", "similarity_score", "image_path"}
    return [col for col in df.columns if col not in excluded]


# ATTRIBUTE ANALYSIS

def split_multivalue(value: str) -> List[str]:
    """
    Light splitter for values that may contain multiple labels.
    Adjust separators if your annotation format uses something else.
    """
    candidates = [value]

    for sep in ["|", ",", ";"]:
        next_candidates = []
        for candidate in candidates:
            if sep in candidate:
                next_candidates.extend(candidate.split(sep))
            else:
                next_candidates.append(candidate)
        candidates = next_candidates

    result = []
    for item in candidates:
        cleaned = clean_value(item)
        if cleaned is not None:
            result.append(cleaned)

    return result


def compute_attribute_frequencies(
    df: pd.DataFrame,
    attribute_columns: List[str],
) -> pd.DataFrame:
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
            rows.append(
                {
                    "attribute_name": col,
                    "attribute_value": attr_value,
                    "count": count,
                    "frequency": count / n_items if n_items > 0 else 0.0,
                }
            )

    freq_df = pd.DataFrame(rows)

    if not freq_df.empty:
        freq_df = freq_df.sort_values(
            by=["attribute_name", "count"],
            ascending=[True, False],
        ).reset_index(drop=True)

    return freq_df


def shannon_entropy(probabilities: List[float]) -> float:
    probs = [p for p in probabilities if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


def compute_attribute_entropy(freq_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if freq_df.empty:
        return pd.DataFrame(
            columns=[
                "attribute_name",
                "num_distinct_values",
                "entropy",
                "normalized_entropy",
                "top_value",
                "top_value_frequency",
            ]
        )

    for attr_name, group in freq_df.groupby("attribute_name"):
        probs = group["frequency"].tolist()
        entropy = shannon_entropy(probs)
        num_values = len(group)
        max_entropy = math.log2(num_values) if num_values > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        top_row = group.sort_values("count", ascending=False).iloc[0]

        rows.append(
            {
                "attribute_name": attr_name,
                "num_distinct_values": int(num_values),
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "top_value": top_row["attribute_value"],
                "top_value_frequency": float(top_row["frequency"]),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        by=["top_value_frequency", "normalized_entropy"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return result


# CLUSTERING BY ATTRIBUTES

def build_attribute_documents(
    df: pd.DataFrame,
    attribute_columns: List[str],
    min_token_freq_ratio: float,
) -> tuple[List[str], List[str]]:
    token_lists: List[List[str]] = []

    for _, row in df.iterrows():
        tokens: List[str] = []

        for col in attribute_columns:
            value = clean_value(row[col]) if col in row else None
            if value is None:
                continue

            for part in split_multivalue(value):
                token = f"{col}={part.lower().replace(' ', '_')}"
                tokens.append(token)

        token_lists.append(tokens)

    all_tokens = [token for tokens in token_lists for token in tokens]
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


def cluster_by_attributes(
    df: pd.DataFrame,
    attribute_columns: List[str],
    max_clusters: int,
    min_token_freq_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[int], Optional[float]]:
    docs, kept_tokens = build_attribute_documents(
        df=df,
        attribute_columns=attribute_columns,
        min_token_freq_ratio=min_token_freq_ratio,
    )

    if not kept_tokens:
        clustered_df = df.copy()
        clustered_df["cluster_id"] = 0

        summary_df = pd.DataFrame([{"cluster_id": 0, "size": len(df)}])
        return clustered_df, summary_df, None, None

    usable_mask = [bool(doc.strip()) for doc in docs]
    usable_df = df.loc[usable_mask].copy().reset_index(drop=True)
    usable_docs = [doc for doc in docs if doc.strip()]

    if len(usable_docs) < 3:
        clustered_df = df.copy()
        clustered_df["cluster_id"] = 0

        summary_df = pd.DataFrame([{"cluster_id": 0, "size": len(df)}])
        return clustered_df, summary_df, None, None

    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(usable_docs)

    best_k: Optional[int] = None
    best_score: Optional[float] = None
    best_labels = None

    upper_k = min(max_clusters, len(usable_df) - 1)

    for k in range(2, upper_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X, labels)
            if best_score is None or score > best_score:
                best_score = float(score)
                best_k = k
                best_labels = labels
        except Exception as e:
            print(f"Clustering failed for k={k}: {e}")

    clustered_df = df.copy()
    clustered_df["cluster_id"] = -1

    if best_labels is None:
        clustered_df["cluster_id"] = 0
        summary_df = pd.DataFrame([{"cluster_id": 0, "size": len(df)}])
        return clustered_df, summary_df, None, None

    usable_df["cluster_id"] = best_labels
    clustered_df.loc[usable_mask, "cluster_id"] = usable_df["cluster_id"].values
    clustered_df.loc[~pd.Series(usable_mask).values, "cluster_id"] = -1

    summary_rows = []
    for cluster_id, group in usable_df.groupby("cluster_id"):
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(group)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="size", ascending=False).reset_index(drop=True)

    return clustered_df, summary_df, best_k, best_score


def summarize_cluster_top_attributes(
    clustered_df: pd.DataFrame,
    attribute_columns: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    rows = []

    valid_df = clustered_df[clustered_df["cluster_id"] >= 0].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["cluster_id", "attribute_name", "attribute_value", "count", "frequency_in_cluster"]
        )

    for cluster_id, group in valid_df.groupby("cluster_id"):
        cluster_size = len(group)
        counts: Dict[tuple[str, str], int] = {}

        for col in attribute_columns:
            for raw_value in group[col]:
                value = clean_value(raw_value)
                if value is None:
                    continue

                for part in split_multivalue(value):
                    key = (col, part)
                    counts[key] = counts.get(key, 0) + 1

        top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for (attr_name, attr_value), count in top_items:
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "attribute_name": attr_name,
                    "attribute_value": attr_value,
                    "count": int(count),
                    "frequency_in_cluster": count / cluster_size if cluster_size > 0 else 0.0,
                }
            )

    return pd.DataFrame(rows)



# SAVE

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# MAIN

def run_experiment(query: str, top_k: int) -> None:
    ensure_output_dir()

    fashion_encoder = FashionCLIPEncoder()

    print("Loading catalog...")
    catalog = load_catalog()

    print(f"Encoding query: {query}")
    query_embedding = fashion_encoder.encode_text(query)

    print(f"Retrieving top-{top_k} items...")
    retrieved_df = retrieve_top_k(
        query_embedding=query_embedding,
        image_ids=catalog.image_ids,
        image_embeddings=catalog.embeddings,
        top_k=top_k,
    )

    print("Building metadata table...")
    metadata_df = build_retrieved_metadata_df(
        retrieved_df=retrieved_df,
        catalog=catalog,
    )

    attribute_columns = infer_attribute_columns(metadata_df)
    print(f"Using attribute columns: {attribute_columns}")

    print("Computing attribute frequencies...")
    freq_df = compute_attribute_frequencies(
        df=metadata_df,
        attribute_columns=attribute_columns,
    )

    print("Computing attribute entropy...")
    entropy_df = compute_attribute_entropy(freq_df)

    print("Clustering by attributes...")
    clustered_df, cluster_summary_df, best_k, silhouette = cluster_by_attributes(
        df=metadata_df,
        attribute_columns=attribute_columns,
        max_clusters=MAX_CLUSTERS,
        min_token_freq_ratio=MIN_TOKEN_FREQ_RATIO,
        random_state=RANDOM_STATE,
    )

    print("Summarizing cluster attributes...")
    cluster_attr_df = summarize_cluster_top_attributes(
        clustered_df=clustered_df,
        attribute_columns=attribute_columns,
        top_n=10,
    )

    query_slug = safe_slug(query)

    retrieved_path = OUTPUT_DIR / f"{query_slug}_retrieved.csv"
    metadata_path = OUTPUT_DIR / f"{query_slug}_retrieved_with_attributes.csv"
    freq_path = OUTPUT_DIR / f"{query_slug}_attribute_frequencies.csv"
    entropy_path = OUTPUT_DIR / f"{query_slug}_attribute_entropy.csv"
    clustered_path = OUTPUT_DIR / f"{query_slug}_clustered_items.csv"
    cluster_summary_path = OUTPUT_DIR / f"{query_slug}_cluster_summary.csv"
    cluster_attr_path = OUTPUT_DIR / f"{query_slug}_cluster_top_attributes.csv"
    summary_path = OUTPUT_DIR / f"{query_slug}_summary.json"

    print("Saving outputs...")
    save_dataframe(retrieved_df, retrieved_path)
    save_dataframe(metadata_df, metadata_path)
    save_dataframe(freq_df, freq_path)
    save_dataframe(entropy_df, entropy_path)
    save_dataframe(clustered_df, clustered_path)
    save_dataframe(cluster_summary_df, cluster_summary_path)
    save_dataframe(cluster_attr_df, cluster_attr_path)

    summary = {
        "query": query,
        "top_k": top_k,
        "num_retrieved": int(len(retrieved_df)),
        "num_attribute_columns": int(len(attribute_columns)),
        "attribute_columns": attribute_columns,
        "best_num_clusters": best_k,
        "silhouette_score": silhouette,
        "output_files": {
            "retrieved": str(retrieved_path),
            "retrieved_with_attributes": str(metadata_path),
            "attribute_frequencies": str(freq_path),
            "attribute_entropy": str(entropy_path),
            "clustered_items": str(clustered_path),
            "cluster_summary": str(cluster_summary_path),
            "cluster_top_attributes": str(cluster_attr_path),
        },
    }
    save_json(summary, summary_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Retrieved items: {len(retrieved_df)}")
    print(f"Attribute columns: {len(attribute_columns)}")
    print(f"Best number of clusters: {best_k}")
    print(f"Silhouette score: {silhouette}")

    print("\nTop dominant attributes:")
    if entropy_df.empty:
        print("No attribute statistics found.")
    else:
        for _, row in entropy_df.head(10).iterrows():
            print(
                f"  {row['attribute_name']}: "
                f"top='{row['top_value']}', "
                f"top_freq={row['top_value_frequency']:.3f}, "
                f"norm_entropy={row['normalized_entropy']:.3f}"
            )

    print("\nCluster sizes:")
    if cluster_summary_df.empty:
        print("No clusters found.")
    else:
        for _, row in cluster_summary_df.iterrows():
            print(f"  cluster={row['cluster_id']}, size={row['size']}")

    print("\nSaved to:")
    print(f"  {OUTPUT_DIR}")


run_experiment(query=QUERY, top_k=TOP_K)