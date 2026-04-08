import type { ChatRequest, ChatResponse } from "@/types";

/**
 * All fetch calls go to Next.js API routes (/api/chat, /api/reset)
 * which proxy to the Python backend — no CORS issues in the browser.
 */

export async function sendChatMessage(
  body: ChatRequest
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Chat API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<ChatResponse>;
}

export async function resetChat(): Promise<void> {
  const res = await fetch("/api/reset", { method: "DELETE" });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Reset API error ${res.status}: ${text}`);
  }
}
