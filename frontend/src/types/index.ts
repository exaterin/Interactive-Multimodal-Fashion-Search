// ── Domain types ──────────────────────────────────────────────────────────────

export interface LikedItem {
  id: string;
  category?: string;
  attributes?: Record<string, string[]>;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Refinement suggestions attached to this assistant turn */
  suggestions?: string[];
  /** Image URLs to show as thumbnails inside a user message (liked items) */
  likedImages?: string[];
  timestamp: Date;
}

export interface Product {
  id: string;
  /** Absolute URL served by the Python backend, e.g. http://localhost:8000/images/<id> */
  image_url: string;
  category?: string;
  score?: number;
  /** Key → list of attribute values, e.g. { "silhouette": ["fitted"] } */
  attributes?: Record<string, string[]>;
}

export interface SearchState {
  original_query: string;
  current_query: string;
  positive_constraints: string[];
  negative_constraints: string[];
  style_tags: string[];
  occasion: string;
}

// ── API contract ──────────────────────────────────────────────────────────────

/** Body sent to POST /api/chat */
export interface ChatRequest {
  message: string;
  search_state: SearchState;
  liked_items?: LikedItem[];
  use_image_similarity?: boolean;
  grounding_mode?: "attribute" | "description" | "image";
}

/** Shape returned by POST /api/chat */
export interface ChatResponse {
  message: string;
  suggestions: string[];
  products: Product[];
  search_state: SearchState;
}
