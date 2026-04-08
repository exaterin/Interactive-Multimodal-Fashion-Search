import { NextRequest, NextResponse } from "next/server";

/**
 * POST /api/chat
 * Proxies to the Python backend at BACKEND_URL/chat.
 *
 * Expected Python backend contract:
 *   POST /chat
 *   Body: { message: string, search_state: SearchState }
 *   Returns: { message, suggestions, products, search_state }
 */
export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  try {
    const body = await req.json();

    const upstream = await fetch(`${backendUrl}/chat`, {
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
