// ── Domain types ──────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Refinement suggestions attached to this assistant turn */
  suggestions?: string[];
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
}

/** Shape returned by POST /api/chat */
export interface ChatResponse {
  message: string;
  suggestions: string[];
  products: Product[];
  search_state: SearchState;
}
