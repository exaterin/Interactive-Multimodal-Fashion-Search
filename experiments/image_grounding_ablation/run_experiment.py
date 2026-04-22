"""
Image grounding capability experiment.

For each query and each input size, retrieves the top N
results with FashionCLIP, then sends ALL N images in a single multimodal LLM call
and records the structured summary.

This tests raw LLM capability at different context sizes.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation.llm_client import LLMClient
from src.data.fashionpedia.loaders import load_fashionpedia_catalog
from src.models.fashion_clip_encoder import build_fashion_clip_encoder
from src.retrieval.fashionpedia_retriever import search_clip_fp

# ── Config ────────────────────────────────────────────────────────────────────

QUERIES: List[str] = [
    "floral summer dress",
    "oversized denim jacket",
    "formal black suit",
    "casual white t-shirt",
    "elegant evening gown",
    "striped linen shirt",
    "leather ankle boots",
]

INPUT_SIZES: List[int] = [10, 20, 35, 50, 70, 100, 150, 200, 250]

THUMB_SIZE = (224, 224)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "image_grounding_ablation"

SYSTEM_PROMPT = """\
You are a fashion visual analyst. You will be shown a set of fashion item images \
retrieved for a specific search query.
Analyze ALL images together and return a single structured JSON summary describing \
what you observe across the entire set.

Return ONLY valid JSON with this exact schema:
{
  "dominant_item_types": ["string", ...],
  "dominant_colors": ["string", ...],
  "common_materials_or_textures": ["string", ...],
  "recurring_patterns_or_details": ["string", ...],
  "overall_style_or_aesthetic": "string",
  "notable_differences_or_outliers": "string"
  "summary_description": "string"
}
"""

# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image_b64(catalog, item_id: str) -> Optional[str]:
    image_path = catalog.image_paths.get(item_id)
    if not image_path or not image_path.exists():
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        bbox = catalog.bboxes.get(item_id)
        if bbox:
            x, y, w, h = (int(v) for v in bbox)
            img = img.crop((x, y, x + w, y + h))
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _load_images(catalog, results: List[dict]) -> List[Tuple[str, str]]:
    """Return (item_id, base64_jpeg) pairs for items that load successfully."""
    loaded = []
    for item in results:
        item_id = item["image_id"]
        b64 = _load_image_b64(catalog, item_id)
        if b64:
            loaded.append((item_id, b64))
    return loaded

# ── Single-call LLM grounding ─────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def analyze_with_single_call(
    results: List[dict],
    catalog,
    llm_client: LLMClient,
    query: str,
    input_size: int,
) -> Tuple[Optional[dict], str]:
    """
    Send all images in one multimodal LLM call.
    Returns (parsed_summary_dict, raw_response_text).
    """
    images = _load_images(catalog, results)
    n_loaded = len(images)
    print(f"    Loaded {n_loaded}/{len(results)} images")

    content: list = [
        {
            "type": "text",
            "text": (
                f"Query: \"{query}\"\n"
                f"I retrieved {n_loaded} fashion items for this query. "
                f"Analyze all {n_loaded} images below and return a structured summary."
            ),
        }
    ]
    for _, b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    raw = llm_client.chat(messages)
    parsed = _parse_json(raw)
    return parsed, raw

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_slug(text: str) -> str:
    return (
        text.lower().strip()
        .replace(" ", "_").replace("/", "_").replace("\\", "_")
        .replace(",", "").replace(".", "")
    )


def print_summary(parsed: dict, size: int, elapsed: float) -> None:
    W = 70
    print(f"\n    {'─'*W}")
    print(f"    Input size: {size} images  |  ⏱ {elapsed:.1f}s")
    print(f"    {'─'*W}")
    if parsed.get("dominant_item_types"):
        print(f"    Item types : {', '.join(parsed['dominant_item_types'])}")
    if parsed.get("dominant_colors"):
        print(f"    Colors     : {', '.join(parsed['dominant_colors'])}")
    if parsed.get("common_materials_or_textures"):
        print(f"    Materials  : {', '.join(parsed['common_materials_or_textures'])}")
    if parsed.get("recurring_patterns_or_details"):
        print(f"    Patterns   : {', '.join(parsed['recurring_patterns_or_details'])}")
    if parsed.get("overall_style_or_aesthetic"):
        print(f"    Style      : {parsed['overall_style_or_aesthetic']}")
    if parsed.get("notable_differences_or_outliers"):
        print(f"    Outliers   : {parsed['notable_differences_or_outliers']}")


def build_markdown_summary(all_results: List[dict]) -> str:
    lines: List[str] = [
        "# Image Grounding Capability Experiment",
        "",
        f"**Mode**: single LLM call per input size (no batching)  ",
        f"**Queries**: {len(all_results)}  ",
        f"**Input sizes**: {INPUT_SIZES}",
        "",
    ]

    for entry in all_results:
        query = entry["query"]
        lines += [f"## {query}", ""]

        for size in INPUT_SIZES:
            r = entry["results"].get(str(size))
            if not r:
                continue
            elapsed = r.get("elapsed_seconds", 0)
            lines.append(f"### {size} images  _(⏱ {elapsed:.1f}s)_")
            lines.append("")

            if r.get("error"):
                lines += [f"**Error**: {r['error']}", ""]
                continue

            ctx = r.get("grounding") or {}
            lines += [
                f"- **Item types**: {', '.join(ctx.get('dominant_item_types', [])) or '—'}",
                f"- **Colors**: {', '.join(ctx.get('dominant_colors', [])) or '—'}",
                f"- **Materials**: {', '.join(ctx.get('common_materials_or_textures', [])) or '—'}",
                f"- **Patterns**: {', '.join(ctx.get('recurring_patterns_or_details', [])) or '—'}",
                f"- **Style**: {ctx.get('overall_style_or_aesthetic') or '—'}",
                f"- **Outliers**: {ctx.get('notable_differences_or_outliers') or '—'}",
                "",
            ]

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fashionpedia catalog…")
    catalog = load_fashionpedia_catalog()
    print(f"  {len(catalog.item_ids):,} items loaded")

    print("Loading FashionCLIP encoder…")
    encoder = build_fashion_clip_encoder()

    llm_client = LLMClient(
        model="google/gemini-3-flash-preview",
        temperature=0.0,
        timeout=180,
    )

    all_results: List[dict] = []

    for query_idx, query in enumerate(QUERIES):
        slug = safe_slug(query)
        W = 72
        print(f"\n{'='*W}")
        print(f"Query {query_idx + 1}/{len(QUERIES)}: '{query}'")
        print(f"{'='*W}")

        # Retrieve top 50 once, reuse slices per input size
        max_size = max(INPUT_SIZES)
        print(f"  Retrieving top {max_size} items…")
        results = search_clip_fp(
            catalog=catalog,
            encoder=encoder,
            query_text=query,
            top_k=max_size,
        )
        print(f"  Retrieved {len(results)} items")

        query_entry: dict = {
            "query": query,
            "slug": slug,
            "num_retrieved": len(results),
            "results": {},
        }

        for size in INPUT_SIZES:
            print(f"\n  ── input_size={size} ──")
            t0 = time.perf_counter()

            try:
                parsed, raw = analyze_with_single_call(
                    results=results[:size],
                    catalog=catalog,
                    llm_client=llm_client,
                    query=query,
                    input_size=size,
                )
                elapsed = time.perf_counter() - t0

                if parsed:
                    print_summary(parsed, size, elapsed)
                    query_entry["results"][str(size)] = {
                        "input_size": size,
                        "elapsed_seconds": round(elapsed, 2),
                        "grounding": parsed,
                        "raw_response": raw,
                    }
                else:
                    print(f"    WARNING: LLM returned unparseable JSON (size={size})")
                    query_entry["results"][str(size)] = {
                        "input_size": size,
                        "elapsed_seconds": round(elapsed, 2),
                        "grounding": None,
                        "raw_response": raw,
                        "error": "json_parse_failed",
                    }

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"    ERROR: {exc}")
                query_entry["results"][str(size)] = {
                    "input_size": size,
                    "elapsed_seconds": round(elapsed, 2),
                    "grounding": None,
                    "raw_response": None,
                    "error": str(exc),
                }

        all_results.append(query_entry)

        # Save per-query file immediately (recoverable on partial run)
        query_path = OUTPUT_DIR / f"{slug}.json"
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(query_entry, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved → {query_path.relative_to(PROJECT_ROOT)}")

    # ── Summary files ──────────────────────────────────────────────────────────

    comparison_rows: List[dict] = []
    for entry in all_results:
        for size in INPUT_SIZES:
            r = entry["results"].get(str(size))
            if not r:
                continue
            ctx = r.get("grounding") or {}
            comparison_rows.append({
                "query": entry["query"],
                "input_size": size,
                "elapsed_seconds": r.get("elapsed_seconds"),
                "status": "error" if r.get("error") else "ok",
                "num_item_types": len(ctx.get("dominant_item_types", [])),
                "num_colors": len(ctx.get("dominant_colors", [])),
                "num_materials": len(ctx.get("common_materials_or_textures", [])),
                "num_patterns": len(ctx.get("recurring_patterns_or_details", [])),
                "has_style": bool(ctx.get("overall_style_or_aesthetic")),
                "has_outliers": bool(ctx.get("notable_differences_or_outliers")),
            })

    summary = {
        "mode": "single_call_no_batching",
        "queries": QUERIES,
        "input_sizes": INPUT_SIZES,
        "comparison_table": comparison_rows,
        "per_query": all_results,
    }

    summary_json_path = OUTPUT_DIR / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_md_path = OUTPUT_DIR / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_summary(all_results))

    # ── Timing table ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Experiment complete.")
    print(f"  summary.json → {summary_json_path.relative_to(PROJECT_ROOT)}")
    print(f"  summary.md   → {summary_md_path.relative_to(PROJECT_ROOT)}")
    print(f"\n{'Timing (seconds)':^72}")
    print(f"{'─'*72}")
    header = f"  {'Query':<32}" + "".join(f"  {s:>5}img" for s in INPUT_SIZES)
    print(header)
    print(f"  {'─'*32}" + "".join(f"  {'─'*6}" for _ in INPUT_SIZES))
    for entry in all_results:
        row = f"  {entry['query'][:32]:<32}"
        for size in INPUT_SIZES:
            r = entry["results"].get(str(size))
            if r and not r.get("error"):
                row += f"  {r['elapsed_seconds']:>5.1f}s"
            else:
                row += f"  {'ERR':>6}"
        print(row)
    print(f"{'─'*72}")


if __name__ == "__main__":
    run()
