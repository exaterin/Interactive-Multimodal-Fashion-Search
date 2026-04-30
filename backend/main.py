"""
FastAPI backend for the Fashion Search chatbot.

Two-LLM pipeline per /chat call (matches the project schema):
  1. Query Composer LLM (#1)                 → query (+ reset / new_query flags)
  2. Multimodal Retriever #1 (Composer query) → grounding
  3. Catalog Evidence                         — Context Extraction over results
  4. Preference Evidence                      — liked items, same Context Extraction
  5. Response Generator LLM (#2)              → response, suggestions, updated_query,
                                                 intent, constraints
  6. Multimodal Retriever #2 (updated_query)  → items shown in the UI
                                                 (skipped when updated_query is unchanged)

Endpoints:
  POST   /chat          — runs the pipeline above
  DELETE /reset         — no-op (state lives in the frontend)
  GET    /images/{id}   — serve a bbox-cropped Fashionpedia image as JPEG
"""

from __future__ import annotations

import asyncio
import functools
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation.llm_client import LLMClient
from src.data.fashionpedia.loaders import load_fashionpedia_catalog
from src.models.fashion_clip_encoder import build_fashion_clip_encoder
from src.retrieval.fashionpedia_retriever import search_clip_fp
from src.search.catalog_evidence import build_catalog_evidence
from src.search.context_extraction import parse_strategy
from src.search.preference_evidence import PreferenceItem, build_preference_evidence
from src.search.query_composer import compose_query
from src.search.relevance_feedback import run_relevance_feedback
from src.search.response_generator import generate_grounded_response
from src.search.search_state import SearchState as _SearchState
import src.log as log


# ── Pydantic models (API contract) ────────────────────────────────────────────

class SearchStateSchema(BaseModel):
    original_query: str = ""
    current_query: str = ""
    positive_constraints: List[str] = []
    negative_constraints: List[str] = []
    style_tags: List[str] = []
    occasion: str = ""
    budget: str = ""


class LikedItemSchema(BaseModel):
    id: str
    category: Optional[str] = None
    attributes: Optional[Dict[str, List[str]]] = None


class HistoryMessageSchema(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    search_state: SearchStateSchema
    liked_items: List[LikedItemSchema] = []
    grounding_mode: str = "attribute"
    chat_history: List[HistoryMessageSchema] = []


class FeedbackRequest(BaseModel):
    selected_items: List[LikedItemSchema]
    comment: str = ""
    search_state: SearchStateSchema
    grounding_mode: str = "attribute"
    chat_history: List[HistoryMessageSchema] = []


class ProductSchema(BaseModel):
    id: str
    image_url: str
    category: Optional[str] = None
    score: Optional[float] = None
    attributes: Optional[Dict[str, List[str]]] = None


class ChatResponseSchema(BaseModel):
    message: str
    suggestions: List[str]
    products: List[ProductSchema]
    search_state: SearchStateSchema
    intent: str = ""


# ── State converters ──────────────────────────────────────────────────────────

def _to_dataclass(schema: SearchStateSchema) -> _SearchState:
    s = _SearchState()
    s.original_query = schema.original_query
    s.current_query = schema.current_query
    s.positive_constraints = list(schema.positive_constraints)
    s.negative_constraints = list(schema.negative_constraints)
    s.style_tags = list(schema.style_tags)
    s.occasion = schema.occasion
    s.budget = schema.budget
    return s


def _from_dataclass(s: _SearchState) -> SearchStateSchema:
    return SearchStateSchema(
        original_query=s.original_query,
        current_query=s.current_query,
        positive_constraints=s.positive_constraints,
        negative_constraints=s.negative_constraints,
        style_tags=s.style_tags,
        occasion=s.occasion,
        budget=s.budget,
    )


# ── App startup / shutdown ────────────────────────────────────────────────────

_catalog = None
_encoder = None
_llm_client = None
_thread_pool = ThreadPoolExecutor(max_workers=8)


@functools.lru_cache(maxsize=2000)
def _load_and_crop_image(item_id: str, image_path_str: str, bbox_tuple: Optional[tuple]) -> bytes:
    """Load, crop, and JPEG-encode an image. Result is cached in-process."""
    img = Image.open(image_path_str).convert("RGB")
    if bbox_tuple:
        x, y, w, h = (int(v) for v in bbox_tuple)
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width, x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)
        img = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _catalog, _encoder, _llm_client

    print("Loading Fashionpedia catalog…")
    _catalog = load_fashionpedia_catalog()

    print("Loading FashionCLIP encoder…")
    _encoder = build_fashion_clip_encoder()

    _llm_client = LLMClient(
        model="google/gemini-3-flash-preview",
        temperature=0.0,
        timeout=60,
    )

    print(f"Backend ready. {len(_catalog.item_ids):,} items loaded.")
    yield


app = FastAPI(title="Fashion Search API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _to_preference_items(liked_items: List[LikedItemSchema]) -> List[PreferenceItem]:
    return [
        PreferenceItem(
            id=item.id,
            category=item.category or "",
            attributes=dict(item.attributes or {}),
        )
        for item in liked_items
    ]


@app.post("/chat", response_model=ChatResponseSchema)
async def chat(req: ChatRequest) -> ChatResponseSchema:
    search_state = _to_dataclass(req.search_state)
    history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    strategy = parse_strategy(req.grounding_mode)

    log.turn_start(req.message)
    log.search_state(search_state)
    log.chat_history(history)

    # 1. Query Composer LLM (#1) — produces the retrieval query (+ reset / new_query flags).
    #    Sees only state + user message + history; runs BEFORE retrieval.
    composed = compose_query(
        user_message=req.message,
        search_state=search_state,
        llm_client=_llm_client,
        chat_history=history,
    )

    # 1a. Reset short-circuit — no retrieval, no responder LLM
    if composed.reset:
        search_state.reset()
        log.state_update("", "", search_state)
        log.turn_end()
        return ChatResponseSchema(
            message="",
            suggestions=[],
            products=[],
            search_state=_from_dataclass(search_state),
            intent="reset",
        )

    # 1b. Topic switch — wipe state so the responder builds fresh constraints
    if composed.new_query:
        search_state.reset()

    composer_query = composed.query or search_state.current_query or req.message

    # 2. Multimodal Retriever #1 — uses the Composer's query
    log.retrieval(composer_query, 1000)
    results = search_clip_fp(
        catalog=_catalog,
        encoder=_encoder,
        query_text=composer_query,
        top_k=1000,
    )
    log.retrieval_done(len(results))

    # 3. Catalog Evidence (over Retriever #1)
    catalog_evidence = build_catalog_evidence(results, _catalog, strategy=strategy)
    log.catalog_evidence(catalog_evidence)

    # 4. Preference Evidence
    preference_items = _to_preference_items(req.liked_items)
    preference_evidence = build_preference_evidence(preference_items, _catalog, strategy=strategy)
    log.preference_evidence(preference_evidence)

    # 5. Response Generator LLM (#2) — sees catalog evidence; emits updated_query + structured state
    response_text, suggestions, updated_query, llm_data = generate_grounded_response(
        user_message=req.message,
        search_state=search_state,
        catalog_evidence=catalog_evidence,
        preference_evidence=preference_evidence,
        llm_client=_llm_client,
        chat_history=history,
    )
    intent = llm_data.get("intent", "") or ("new_query" if composed.new_query else "")

    # 6. Update Search State
    if not search_state.original_query:
        search_state.original_query = req.message
    search_state.current_query = updated_query or composer_query
    search_state.last_suggestions = suggestions
    search_state.update_from_llm(llm_data)
    log.state_update(composer_query, search_state.current_query, search_state)

    # 7. Multimodal Retriever #2 — only if the responder refined the query
    if updated_query and updated_query != composer_query:
        log.reretrieval(updated_query)
        results = search_clip_fp(
            catalog=_catalog,
            encoder=_encoder,
            query_text=updated_query,
            top_k=1000,
        )
        log.reretrieval_done(len(results))

    log.turn_end()

    # 8. Build product list for the UI
    products: List[ProductSchema] = []
    for item in results:
        item_id: str = item["image_id"]

        raw_attrs = _catalog.attribute_annotations.get(item_id, {})
        attributes: Dict[str, List[str]] = {
            k: list(v) for k, v in raw_attrs.items() if v
        }
        colors = _catalog.color_annotations.get(item_id, [])
        if colors:
            attributes["color"] = colors

        products.append(
            ProductSchema(
                id=item_id,
                image_url=f"/images/{item_id}",
                category=_catalog.category_annotations.get(item_id),
                score=item.get("score"),
                attributes=attributes or None,
            )
        )

    return ChatResponseSchema(
        message=response_text,
        suggestions=suggestions,
        products=products,
        search_state=_from_dataclass(search_state),
        intent=intent,
    )


@app.post("/feedback", response_model=ChatResponseSchema)
async def feedback(req: FeedbackRequest) -> ChatResponseSchema:
    """
    Relevance feedback pipeline.

    Runs ONLY when the user has selected 1–3 items from the current retrieval
    and (optionally) attached a short comment. Bypasses the standard
    composer→responder /chat flow:

      selected_items + comment + state
        → relevance-feedback LLM (refined_query + response + state)
        → multimodal retrieval
        → UI
    """
    if not req.selected_items:
        raise HTTPException(status_code=400, detail="At least one selected item is required")
    if len(req.selected_items) > 3:
        raise HTTPException(status_code=400, detail="At most 3 items may be selected")

    search_state = _to_dataclass(req.search_state)
    history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    strategy = parse_strategy(req.grounding_mode)
    selected = _to_preference_items(req.selected_items)

    log.turn_start(f"[relevance feedback] {len(selected)} selected | comment: \"{req.comment}\"")
    log.search_state(search_state)
    log.chat_history(history)

    # 1. Relevance Feedback LLM — refined_query + response + state updates
    result = run_relevance_feedback(
        selected_items=selected,
        comment=req.comment,
        search_state=search_state,
        catalog=_catalog,
        llm_client=_llm_client,
        strategy=strategy,
        chat_history=history,
    )

    # 2. Update search state from the LLM output
    if not search_state.original_query:
        search_state.original_query = req.comment or search_state.current_query
    old_query = search_state.current_query
    search_state.current_query = result.refined_query or search_state.current_query
    search_state.update_from_llm(result.raw)
    search_state.last_suggestions = result.suggestions
    log.state_update(old_query, search_state.current_query, search_state)

    # 3. Retrieval with the refined query
    retrieval_query = search_state.current_query or old_query
    log.retrieval(retrieval_query, 1000)
    results = search_clip_fp(
        catalog=_catalog,
        encoder=_encoder,
        query_text=retrieval_query,
        top_k=1000,
    )
    log.retrieval_done(len(results))
    log.turn_end()

    # 4. Build product list
    products: List[ProductSchema] = []
    for item in results:
        item_id: str = item["image_id"]
        raw_attrs = _catalog.attribute_annotations.get(item_id, {})
        attributes: Dict[str, List[str]] = {
            k: list(v) for k, v in raw_attrs.items() if v
        }
        colors = _catalog.color_annotations.get(item_id, [])
        if colors:
            attributes["color"] = colors

        products.append(
            ProductSchema(
                id=item_id,
                image_url=f"/images/{item_id}",
                category=_catalog.category_annotations.get(item_id),
                score=item.get("score"),
                attributes=attributes or None,
            )
        )

    return ChatResponseSchema(
        message=result.response,
        suggestions=result.suggestions,
        products=products,
        search_state=_from_dataclass(search_state),
        intent="relevance_feedback",
    )


@app.delete("/reset", status_code=204)
async def reset() -> None:
    """State is managed on the frontend; this is a no-op hook for cleanup."""


@app.get("/images/{item_id}")
async def get_image(item_id: str) -> Response:
    """Serve a bbox-cropped Fashionpedia image as JPEG (LRU-cached + thread-offloaded)."""
    if _catalog is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded yet")

    image_path = _catalog.image_paths.get(item_id)
    bbox = _catalog.bboxes.get(item_id)

    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {item_id}")

    bbox_tuple = tuple(bbox) if bbox else None

    try:
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(
            _thread_pool,
            _load_and_crop_image,
            item_id,
            str(image_path),
            bbox_tuple,
        )
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=str(Path(__file__).parent),
    )
