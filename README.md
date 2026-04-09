# Interactive Multimodal Fashion Search

An interactive conversational fashion search system that combines multimodal embeddings, semantic retrieval, and large language models to enable natural-language-driven exploration of fashion catalogs.

---

## Overview

The system allows users to search for clothing items through a chat interface using natural language queries such as "elegant summer dress" or "casual blue jeans with minimal pattern." Rather than relying on keyword matching or predefined filters, the system encodes both queries and catalog images into a shared embedding space using [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) and retrieves semantically similar items. A LLM then interprets the user's intent, grounds its response in the actual retrieved results, refines the search query when needed, and maintains conversation state across multiple turns.

This approach implements a multimodal Retrieval-Augmented Generation (RAG) pipeline: instead of generating responses solely from model parameters, the LLM is grounded in factual data retrieved from the catalog at inference time.

---

## Main Functionality

### Conversational Search
- Users interact through a chat interface with multi-turn dialogue support
- The system maintains conversation state across turns: positive/negative constraints, style tags, occasion, and category
- User can say things like "make it more casual" or "no leather" and the system updates constraints incrementally

### Multimodal Semantic Retrieval
- Queries are encoded into a 512-dimensional embedding using FashionCLIP
- Catalog items (bounding-box crops from Fashionpedia annotations) are pre-encoded and stored as `.npy` files
- Retrieval is performed via cosine similarity over the full embedding matrix: `scores = embeddings @ query_embedding`
- Top-k results are returned ranked by similarity score

### LLM-Grounded Response Generation
- After retrieval, the system analyzes result distributions: dominant categories, colors, and attributes
- This grounding context (what was actually found) is passed to the LLM alongside the user message
- The LLM generates a structured JSON response including: a natural-language reply, clickable refinement suggestions, a refined query, and updated constraints
- The LLM is explicitly instructed not to invent items — it must base all suggestions on the retrieved results

### Query Refinement and Re-Retrieval
- If the LLM produces an updated query, the system performs a second retrieval pass before responding
- This enables iterative narrowing: the query evolves through the conversation

### Liked Items Integration
- Users can like items in the results grid
- Liked items are sent to the backend on each turn and included in the LLM prompt as preference signals

### Image Serving
- Catalog items are bounding-box crops from full images (Fashionpedia COCO format)
- The backend serves cropped images dynamically via `GET /images/{item_id}` with LRU caching

---

## Architecture

```
User (browser)
    ↓
Next.js Frontend (React + TypeScript)
    ↓ POST /api/chat
FastAPI Backend (Python)
    ├── 1. CLIP Retrieval — top-k results by cosine similarity
    ├── 2. Grounding Analysis — extract dominant categories/colors/attributes
    ├── 3. LLM Call (via OpenRouter) — generate grounded JSON response
    ├── 4. Re-retrieval (if query was refined by LLM)
    └── 5. Return products + updated state to frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding model | FashionCLIP (vision-language, HuggingFace) |
| LLM | google/gemini-flash-3 via OpenRouter API |
| Backend | FastAPI + Uvicorn (Python) |
| Frontend | Next.js 15, React 19, TypeScript |
| Styling | Tailwind CSS, Framer Motion, Radix UI |
| Dataset | Fashionpedia (COCO-format annotations) |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Fashionpedia dataset (images + COCO JSON annotations)
- OpenRouter API key

### 1. Clone and install Python dependencies

```bash
git clone <repo-url>
cd Interactive-Multimodal-Fashion-Search
python -m venv fashionenv
source fashionenv/bin/activate
pip install fastapi uvicorn fashion-clip pillow numpy torch requests python-dotenv pydantic
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
```

Create `frontend/.env.local`:

```
BACKEND_URL=http://localhost:8000
```

### 3. Prepare the dataset

Download the [Fashionpedia](https://fashionpedia.github.io/home/index.html) training set and place files as:

```
data/fashionpedia/instances_attributes_train2020.json
data/fashionpedia/train/          # image files
data/fashionpedia/color_ann.json  # color annotations
```

### 4. Generate embeddings

Run once to pre-compute FashionCLIP embeddings for all catalog items:

```bash
python src/models/generate_fashionpedia_embeddings.py
```

This produces:
- `data/fashionpedia/embeddings/fashionpedia_embeddings.npy` — (N, 512) float32 matrix
- `data/fashionpedia/embeddings/fashionpedia_item_ids.npy` — aligned item ID array

### 5. Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On startup, the backend loads the full catalog (~50k items), the FashionCLIP model, and the LLM client.

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## API

### `POST /chat`

Main search endpoint. Accepts a user message, current search state, and liked items; returns LLM response, product results, and updated state.

**Request:**
```json
{
  "message": "show me casual blue shirts",
  "search_state": {
    "original_query": "",
    "current_query": "",
    "positive_constraints": [],
    "negative_constraints": [],
    "style_tags": [],
    "occasion": ""
  },
  "liked_items": []
}
```

**Response:**
```json
{
  "message": "Here are some casual blue shirts from the catalog...",
  "suggestions": ["fitted cut", "striped pattern", "short sleeve"],
  "products": [
    {
      "id": "12345",
      "image_url": "/images/12345",
      "category": "shirt, blouse",
      "score": 0.91,
      "attributes": { "neckline type": ["collar"], "length": ["hip length"] }
    }
  ],
  "search_state": { ... }
}
```

### `GET /images/{item_id}`

Returns a JPEG-encoded bounding-box crop of the specified catalog item. LRU-cached (up to 2000 entries).

### `DELETE /reset`

Resets server-side state (conversation state is primarily managed on the frontend).

---

## How It Works

### Retrieval

Text queries are encoded with FashionCLIP's text encoder into a normalized 512-dimensional vector. Cosine similarity is computed against all pre-encoded catalog embeddings:

```
similarity(q, xᵢ) = (q · xᵢ) / (‖q‖ ‖xᵢ‖)
```

The top-k items are returned ranked by score.

### Grounding Analysis

The retrieved results are analyzed to extract statistical summaries:
- Dominant garment categories (e.g., shirt, pants)
- Color distribution
- Dominant attributes per supercategory (silhouette, neckline, pattern, length, etc.)

These are passed to the LLM as factual context about what the catalog actually contains.

### LLM Response Generation

The LLM receives:
1. A system prompt defining its role and output format
2. The current search state (constraints, style tags, occasion)
3. The grounding context (what was retrieved)
4. Any liked items
5. The user's message

It returns structured JSON with a natural-language response, refinement suggestions, an updated query, and updated constraints. If the query changes, re-retrieval is triggered automatically.

---

## Dataset

The system uses the [Fashionpedia](https://fashionpedia.github.io/home/index.html) dataset.
