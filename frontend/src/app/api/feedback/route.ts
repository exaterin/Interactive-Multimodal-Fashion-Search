import { NextRequest, NextResponse } from "next/server";

/**
 * POST /api/feedback
 * Proxies to the Python backend at BACKEND_URL/feedback.
 *
 * Body: { selected_items, comment, search_state, grounding_mode, chat_history }
 * Returns: { message, suggestions, products, search_state, intent }
 */
export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  try {
    const body = await req.json();

    const upstream = await fetch(`${backendUrl}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await upstream.json();

    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: `Backend unreachable: ${message}` },
      { status: 502 }
    );
  }
}
