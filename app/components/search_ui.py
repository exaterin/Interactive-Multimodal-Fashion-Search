from __future__ import annotations

import streamlit as st
from PIL import Image

from src.data.deepfashion.loaders import load_catalog
from src.retrieval.attribute_retriever import search_by_attributes
from src.retrieval.clip_retriever import search_clip
from src.retrieval.hybrid_retriever import search_hybrid

# Filters

def _render_df_filters(catalog):
    return {
        "fabric": st.multiselect("Fabric", options=catalog.available_values["fabric"], default=[]),
        "pattern": st.multiselect("Pattern", options=catalog.available_values["pattern"], default=[]),
        "shape": st.multiselect("Shape", options=catalog.available_values["shape"], default=[]),
    }


def _render_fp_filters(catalog):
    return {
        "category": st.multiselect("Category", options=catalog.available_values["category"], default=[]),
        "pattern": st.multiselect("Pattern", options=catalog.available_values.get("pattern", []), default=[]),
        "shape": st.multiselect("Shape", options=catalog.available_values.get("shape", []), default=[]),
        "material": st.multiselect("Material", options=catalog.available_values.get("material", []), default=[]),
        "finishing": st.multiselect("Finishing", options=catalog.available_values.get("finishing", []), default=[]),
    }


# Result renderers

def _render_df_results(results, catalog) -> None:
    if not results:
        st.info("No results found.")
        return

    st.success(f"Found {len(results)} results.")
    cols_per_row = 3
    for start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, results[start : start + cols_per_row]):
            image_id = item["image_id"]
            image_path = catalog.image_paths[image_id]
            with col:
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning(f"Image not found: {image_id}")
                st.markdown(f"**Rank:** {item['rank']}")
                st.markdown(f"**Image:** `{image_id}`")
                if item["score"] is not None:
                    st.markdown(f"**Cosine similarity:** `{item['score']:.4f}`")


def _render_fp_results(results, catalog) -> None:
    if not results:
        st.info("No results found.")
        return

    st.success(f"Found {len(results)} results.")
    cols_per_row = 3
    for start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, results[start : start + cols_per_row]):
            item_id = item["image_id"]
            image_path = catalog.image_paths.get(item_id)
            bbox = catalog.bboxes.get(item_id)
            with col:
                if image_path and image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning(f"Image not found: {item_id}")
                cat = catalog.category_annotations.get(item_id, "")
                st.markdown(f"**Rank:** {item['rank']}")
                if cat:
                    st.markdown(f"**Category:** {cat}")
                if item["score"] is not None:
                    st.markdown(f"**Cosine similarity:** `{item['score']:.4f}`")


# Main render function

def render_search_tab() -> None:
    st.title("Search")

    # Dataset selector 
    dataset = st.radio("Dataset", ["DeepFashion", "Fashionpedia"], horizontal=True)

    if dataset == "Fashionpedia":
        try:
            from src.data.fashionpedia.loaders import load_fashionpedia_catalog
            catalog = load_fashionpedia_catalog()
        except FileNotFoundError:
            st.error(
                "Fashionpedia embeddings not found. "
                "Run `python src/models/generate_fashionpedia_embeddings.py` first."
            )
            return
    else:
        catalog = load_catalog()

    render_results = _render_fp_results if dataset == "Fashionpedia" else _render_df_results

    # Search type
    search_type = st.radio(
        "Search type",
        options=["CLIP Search", "Attribute Search", "Hybrid Search"],
        horizontal=True,
    )

    top_k = st.slider("Top-K results", min_value=4, max_value=50, value=12, step=2)

    results = []

    # CLIP Search
    if search_type == "CLIP Search":
        st.write("Search by text, image, or both using FashionCLIP.")

        query_text = st.text_input(
            "Text query",
            placeholder="e.g. denim jacket with long sleeves",
        )
        uploaded_file = st.file_uploader(
            "Upload reference image",
            type=["jpg", "jpeg", "png"],
            key="clip_search_upload",
        )
        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Uploaded query image", width=250)

        if st.button("Run CLIP Search", type="primary"):
            if not query_text.strip() and query_image is None:
                st.warning("Please provide text, image, or both.")
                return

            from src.models.fashion_clip_encoder import build_fashion_clip_encoder
            encoder = build_fashion_clip_encoder()

            if dataset == "Fashionpedia":
                from src.retrieval.fashionpedia_retriever import search_clip_fp
                results = search_clip_fp(
                    catalog=catalog,
                    encoder=encoder,
                    query_text=query_text if query_text.strip() else None,
                    query_image=query_image,
                    top_k=top_k,
                )
            else:
                results = search_clip(
                    catalog=catalog,
                    encoder=encoder,
                    query_text=query_text if query_text.strip() else None,
                    query_image=query_image,
                    top_k=top_k,
                )

    # Attribute Search
    elif search_type == "Attribute Search":
        st.write("Search only by structured fashion attributes.")

        if dataset == "Fashionpedia":
            selected_filters = _render_fp_filters(catalog)
        else:
            selected_filters = _render_df_filters(catalog)

        if st.button("Run Attribute Search", type="primary"):
            if not any(selected_filters.values()):
                st.warning("Please select at least one attribute.")
                return

            if dataset == "Fashionpedia":
                from src.retrieval.fashionpedia_retriever import search_by_attributes_fp
                results = search_by_attributes_fp(
                    catalog=catalog,
                    selected_filters=selected_filters,
                    top_k=top_k,
                )
            else:
                results = search_by_attributes(
                    catalog=catalog,
                    selected_filters=selected_filters,
                    top_k=top_k,
                )

    # Hybrid Search
    else:
        st.write("Filter by attributes and rank matching items with FashionCLIP.")

        if dataset == "Fashionpedia":
            selected_filters = _render_fp_filters(catalog)
        else:
            selected_filters = _render_df_filters(catalog)

        query_text = st.text_input(
            "Text query",
            placeholder="e.g. denim jacket with long sleeves",
            key="hybrid_text_query",
        )
        uploaded_file = st.file_uploader(
            "Upload reference image",
            type=["jpg", "jpeg", "png"],
            key="hybrid_search_upload",
        )
        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Uploaded query image", width=250)

        if st.button("Run Hybrid Search", type="primary"):
            if not query_text.strip() and query_image is None:
                st.warning("Please provide text, image, or both.")
                return

            from src.models.fashion_clip_encoder import build_fashion_clip_encoder
            encoder = build_fashion_clip_encoder()

            if dataset == "Fashionpedia":
                from src.retrieval.fashionpedia_retriever import search_hybrid_fp
                results = search_hybrid_fp(
                    catalog=catalog,
                    encoder=encoder,
                    selected_filters=selected_filters,
                    query_text=query_text if query_text.strip() else None,
                    query_image=query_image,
                    top_k=top_k,
                )
            else:
                results = search_hybrid(
                    catalog=catalog,
                    encoder=encoder,
                    selected_filters=selected_filters,
                    query_text=query_text if query_text.strip() else None,
                    query_image=query_image,
                    top_k=top_k,
                )

    render_results(results, catalog)
