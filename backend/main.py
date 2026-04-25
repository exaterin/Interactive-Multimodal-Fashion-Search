"""
FastAPI backend for the Fashion Search chatbot.

Endpoints:
  POST   /chat          — retrieval + grounding + LLM response
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
from src.search.grounding import build_grounding_context, GroundingStrategy
from src.search.relevance_feedback import FeedbackItem
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

def _to_feedback_items(liked_items: List[LikedItemSchema]) -> List[FeedbackItem]:
    return [
        FeedbackItem(
            id=item.id,
            category=item.category or "",
            attributes=dict(item.attributes or {}),
        )
        for item in liked_items
    ]


@app.post("/chat", response_model=ChatResponseSchema)
async def chat(req: ChatRequest) -> ChatResponseSchema:
    search_state = _to_dataclass(req.search_state)

    # Use current_query for retrieval if we already have one; else raw message
    retrieval_query = search_state.current_query or req.message

    history = [{"role": m.role, "content": m.content} for m in req.chat_history]

    log.turn_start(req.message)
    log.search_state(search_state)
    log.chat_history(history)

    # 1. Retrieve
    feedback_items = _to_feedback_items(req.liked_items)
    log.retrieval(retrieval_query, 1000)
    results = search_clip_fp(
        catalog=_catalog,
        encoder=_encoder,
        query_text=retrieval_query,
        top_k=1000,
    )
    log.retrieval_done(len(results))

    # 2. Grounding analysis (feedback folded in when provided)
    strategy = GroundingStrategy(req.grounding_mode) if req.grounding_mode in GroundingStrategy._value2member_map_ else GroundingStrategy.ATTRIBUTE
    grounding = build_grounding_context(results, _catalog, feedback_items=feedback_items, strategy=strategy)
    log.grounding(grounding)
    log.feedback(grounding.feedback_context)

    # 3. LLM response
    response_text, suggestions, updated_query, llm_data = generate_grounded_response(
        user_message=req.message,
        search_state=search_state,
        grounding_context=grounding,
        llm_client=_llm_client,
        chat_history=history,
    )

    # 4. Update state
    intent = llm_data.get("intent", "")
    if intent == "reset":
        search_state.reset()
    else:
        if not search_state.original_query:
            search_state.original_query = req.message
        search_state.current_query = updated_query or retrieval_query
        search_state.last_suggestions = suggestions
        search_state.update_from_llm(llm_data)

    log.state_update(retrieval_query, search_state.current_query, search_state)

    # 5. Re-retrieve with refined text query
    if intent != "reset" and updated_query and updated_query != retrieval_query:
        log.reretrieval(updated_query)
        results = search_clip_fp(
            catalog=_catalog,
            encoder=_encoder,
            query_text=updated_query,
            top_k=1000,
        )
        log.reretrieval_done(len(results))

    log.turn_end()

    # 6. Build product list
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
