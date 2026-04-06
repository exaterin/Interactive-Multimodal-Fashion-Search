from __future__ import annotations

from typing import List, Optional

import streamlit as st
from PIL import Image

from src.conversation.llm_client import LLMClient
from src.search.search_state import SearchState
from src.search.grounding_analyzer import GroundingContext, analyze_results
from src.search.response_generator import generate_grounded_response


# Session state keys
_MSG_KEY = "cb_messages"
_STATE_KEY = "cb_search_state"
_RESULTS_KEY = "cb_results"
_CATALOG_KEY = "cb_catalog"
_SUGGESTIONS_KEY = "cb_suggestions"
_PENDING_KEY = "cb_pending_input"

def _init_state() -> None:
    defaults = {
        _MSG_KEY: [],
        _STATE_KEY: SearchState(),
        _RESULTS_KEY: None,
        _CATALOG_KEY: None,
        _SUGGESTIONS_KEY: [],
        _PENDING_KEY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_all() -> None:
    st.session_state[_MSG_KEY] = []
    st.session_state[_STATE_KEY] = SearchState()
    st.session_state[_RESULTS_KEY] = None
    st.session_state[_CATALOG_KEY] = None
    st.session_state[_SUGGESTIONS_KEY] = []
    st.session_state[_PENDING_KEY] = None

@st.cache_resource
def _load_fashionpedia_catalog():
    from src.data.fashionpedia.loaders import load_fashionpedia_catalog
    return load_fashionpedia_catalog()


@st.cache_resource
def _load_encoder():
    from src.models.fashion_clip_encoder import build_fashion_clip_encoder
    return build_fashion_clip_encoder()

def _crop_bbox(image_path, bbox) -> Image.Image:
    img = Image.open(str(image_path)).convert("RGB")
    x, y, w, h = (int(v) for v in bbox)
    return img.crop((x, y, x + w, y + h))


# Retrieval + grounding + response
def _run_pipeline(
    user_message: str,
    llm_client: LLMClient,
    top_k: int = 12,
) -> tuple[str, List[str]]:
    """
    Run the full pipeline for one user turn:
      1. Build retrieval query from current search state
      2. Retrieve top-K items from Fashionpedia
      3. Analyze retrieved results (grounding)
      4. Generate grounded LLM response + suggestions
      5. Update session state (results, search state, suggestions)

    Returns (response_text, suggestions).
    """
    search_state: SearchState = st.session_state[_STATE_KEY]

    query = search_state.current_query if search_state.current_query else user_message

    # Retrieval
    try:
        catalog = _load_fashionpedia_catalog()
        encoder = _load_encoder()
    except FileNotFoundError as exc:
        return (
            f"Could not load the catalog. Make sure embeddings are generated: `{exc}`",
            [],
        )

    from src.retrieval.fashionpedia_retriever import search_clip_fp

    results = search_clip_fp(
        catalog=catalog,
        encoder=encoder,
        query_text=query,
        top_k=top_k,
    )

    st.session_state[_RESULTS_KEY] = {"results": results, "catalog": catalog}

    # Grounding analysis
    grounding: GroundingContext = analyze_results(results, catalog)

    # LLM response 
    response, suggestions, updated_query, llm_data = generate_grounded_response(
        user_message=user_message,
        search_state=search_state,
        grounding_context=grounding,
        llm_client=llm_client,
    )

    # Update search state 
    if not search_state.original_query:
        search_state.original_query = user_message
    search_state.current_query = updated_query or query
    search_state.last_suggestions = suggestions
    search_state.update_from_llm(llm_data)

    # If the LLM signalled a reset, wipe history and re-init
    if llm_data.get("intent") == "reset":
        _clear_all()
        return "Alright, let's start fresh! What are you looking for?", []

    st.session_state[_STATE_KEY] = search_state
    st.session_state[_SUGGESTIONS_KEY] = suggestions

    # Re-run retrieval with the refined query if it changed meaningfully
    if updated_query and updated_query != query:
        results = search_clip_fp(
            catalog=catalog,
            encoder=encoder,
            query_text=updated_query,
            top_k=top_k,
        )
        st.session_state[_RESULTS_KEY] = {"results": results, "catalog": catalog}

    return response, suggestions


# Result rendering (right panel)

def _render_results_panel() -> None:
    search_state: SearchState = st.session_state[_STATE_KEY]
    results_data = st.session_state[_RESULTS_KEY]

    # Active query summary
    if search_state.current_query:
        st.caption(f"**Query:** {search_state.current_query}")
        constraint_parts = []
        if search_state.positive_constraints:
            constraint_parts.append("✓ " + ", ".join(search_state.positive_constraints))
        if search_state.negative_constraints:
            constraint_parts.append("✗ " + ", ".join(search_state.negative_constraints))
        if search_state.style_tags:
            constraint_parts.append("Style: " + ", ".join(search_state.style_tags))
        if constraint_parts:
            st.caption(" · ".join(constraint_parts))

    if not results_data:
        st.info("Results will appear here once you start a conversation.")
        return

    catalog = results_data["catalog"]
    results: List[dict] = results_data["results"]

    if not results:
        st.warning("No items found for this query.")
        return

    st.markdown(f"*{len(results)} items retrieved*")

    cols_per_row = 3
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
                score = item.get("score")
                if score is not None:
                    st.caption(f"{score:.3f}")


# Chat panel

def _render_chat_panel(llm_client: LLMClient) -> None:
    # Top controls 
    if st.button("Start over", key="cb_clear_btn", type="secondary"):
        _clear_all()
        st.rerun()

    # --- Scrollable chat history ---
    chat_container = st.container(height=520, border=False)
    with chat_container:
        for msg in st.session_state[_MSG_KEY]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- Suggestion buttons (shown between history and input) ---
    suggestions: List[str] = st.session_state[_SUGGESTIONS_KEY]
    if suggestions:
        btn_cols = st.columns(min(len(suggestions), 2))
        for i, sug in enumerate(suggestions):
            with btn_cols[i % 2]:
                if st.button(
                    sug,
                    key=f"cb_sug_{i}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state[_PENDING_KEY] = sug
                    st.rerun()

    # --- Chat input ---
    user_input: Optional[str] = st.chat_input("Describe what you're looking for…")

    # Resolve effective input (typed or from suggestion button)
    pending: Optional[str] = st.session_state.get(_PENDING_KEY)
    effective_input: Optional[str] = pending or user_input
    if pending:
        st.session_state[_PENDING_KEY] = None

    if not effective_input:
        return

    # Append and immediately render user message
    st.session_state[_MSG_KEY].append({"role": "user", "content": effective_input})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(effective_input)

    # Run the retrieval + grounding + LLM pipeline
    with st.spinner("Searching and analyzing…"):
        response_text, _ = _run_pipeline(effective_input, llm_client)

    # Append and render assistant message
    st.session_state[_MSG_KEY].append({"role": "assistant", "content": response_text})
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(response_text)


# ---------------------------------------------------------------------------
# Main entry point (called from streamlit_app.py)
# ---------------------------------------------------------------------------

def render_chat_tab(llm_client: LLMClient) -> None:
    _init_state()

    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        st.subheader("Chat")
        _render_chat_panel(llm_client)

    with right_col:
        st.subheader("Results")
        _render_results_panel()
