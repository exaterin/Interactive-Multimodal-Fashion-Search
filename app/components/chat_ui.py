from __future__ import annotations

from typing import List

import streamlit as st
from PIL import Image

from src.conversation.chat_manager import ChatManager


# Session-state helpers
def init_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    # Last query extracted from LLM reply (or last user message as fallback)
    if "chat_last_query" not in st.session_state:
        st.session_state.chat_last_query = ""
    # Persisted search results: {"results": [...], "dataset": str}
    if "chat_search_panel" not in st.session_state:
        st.session_state.chat_search_panel = None


def clear_chat() -> None:
    st.session_state.chat_messages = []
    st.session_state.chat_last_query = ""
    st.session_state.chat_search_panel = None



# Catalog loaders

def _load_fashionpedia_catalog():
    from src.data.fashionpedia.loaders import load_fashionpedia_catalog
    return load_fashionpedia_catalog()


def _load_deepfashion_catalog():
    from src.data.deepfashion.loaders import load_catalog
    return load_catalog()


# Result renderers

def _crop_bbox(image_path, bbox) -> Image.Image:
    img = Image.open(str(image_path)).convert("RGB")
    x, y, w, h = (int(v) for v in bbox)
    return img.crop((x, y, x + w, y + h))


def _render_fashionpedia_results(results: List[dict], catalog) -> None:
    st.markdown(f"*{len(results)} garments from **Fashionpedia**:*")
    cols_per_row = 4
    for start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, results[start : start + cols_per_row]):
            item_id = item["image_id"]
            image_path = catalog.image_paths.get(item_id)
            bbox = catalog.bboxes.get(item_id)
            with col:
                if image_path and image_path.exists() and bbox:
                    try:
                        st.image(_crop_bbox(image_path, bbox), use_container_width=True)
                    except Exception:
                        st.warning("Image unavailable")
                else:
                    st.warning("Not found")
                cat = catalog.category_annotations.get(item_id, "")
                if cat:
                    st.caption(cat)
                if item.get("score") is not None:
                    st.caption(f"{item['score']:.3f}")


def _render_deepfashion_results(results: List[dict], catalog) -> None:
    st.markdown(f"*{len(results)} items from **DeepFashion**:*")
    cols_per_row = 4
    for start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, results[start : start + cols_per_row]):
            image_id = item["image_id"]
            image_path = catalog.image_paths.get(image_id)
            with col:
                if image_path and image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning("Not found")
                if item.get("score") is not None:
                    st.caption(f"{item['score']:.3f}")



# Search runner

def _run_fashionpedia_search(query: str, top_k: int = 8) -> dict:
    catalog = _load_fashionpedia_catalog()
    from src.models.fashion_clip_encoder import build_fashion_clip_encoder
    from src.retrieval.fashionpedia_retriever import search_clip_fp

    encoder = build_fashion_clip_encoder()
    results = search_clip_fp(catalog=catalog, encoder=encoder, query_text=query, top_k=top_k)
    return {"results": results, "dataset": "Fashionpedia", "catalog": catalog}


def _run_deepfashion_search(query: str, top_k: int = 8) -> dict:
    catalog = _load_deepfashion_catalog()
    from src.models.fashion_clip_encoder import build_fashion_clip_encoder
    from src.retrieval.clip_retriever import search_clip

    encoder = build_fashion_clip_encoder()
    results = search_clip(catalog=catalog, encoder=encoder, query_text=query, top_k=top_k)
    return {"results": results, "dataset": "DeepFashion", "catalog": catalog}


# Main render function

def render_chat_tab(chat_manager: ChatManager) -> None:
    init_chat_state()

    st.title("Chatbot")

    if st.button("Clear chat"):
        clear_chat()
        st.rerun()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Write a message...")

    if user_input:
        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_messages.append(user_message)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        raw_reply = chat_manager.generate_reply(
                            st.session_state.chat_messages
                        )
                    except Exception as exc:
                        raw_reply = f"Error: {exc}"

                display_text, search_query = chat_manager.parse_reply(raw_reply)
                st.markdown(display_text)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": display_text}
        )
        # Store the best available query for the search buttons
        st.session_state.chat_last_query = search_query or user_input

    # Search buttons
    st.divider()

    query = st.session_state.chat_last_query
    if query:
        st.markdown(f"**Search query:** `{query}`")

    col_fp, col_df = st.columns(2)

    with col_fp:
        if st.button("Search in Fashionpedia", use_container_width=True, type="primary"):
            if not query:
                st.warning("Say what you are looking for in the chat first.")
            else:
                with st.spinner(f'Searching Fashionpedia for "{query}"…'):
                    try:
                        panel = _run_fashionpedia_search(query)
                        st.session_state.chat_search_panel = panel
                    except FileNotFoundError:
                        st.error(
                            "Fashionpedia embeddings not found. "
                            "Run `python src/models/generate_fashionpedia_embeddings.py` first."
                        )
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")

    with col_df:
        if st.button("Search in DeepFashion", use_container_width=True):
            if not query:
                st.warning("Say what you are looking for in the chat first.")
            else:
                with st.spinner(f'Searching DeepFashion for "{query}"…'):
                    try:
                        panel = _run_deepfashion_search(query)
                        st.session_state.chat_search_panel = panel
                    except FileNotFoundError:
                        st.error(
                            "DeepFashion embeddings not found. "
                            "Run the embedding generation script first."
                        )
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")

    # Results panel
    panel = st.session_state.chat_search_panel
    if panel:
        catalog = panel["catalog"]
        if panel["dataset"] == "Fashionpedia":
            _render_fashionpedia_results(panel["results"], catalog)
        else:
            _render_deepfashion_results(panel["results"], catalog)
