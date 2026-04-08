import { NextResponse } from "next/server";

/**
 * DELETE /api/reset
 * Proxies to the Python backend at BACKEND_URL/reset to clear server-side
 * conversation state (if any). Returns 204 on success.
 */
export async function DELETE() {
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  try {
    await fetch(`${backendUrl}/reset`, { method: "DELETE" });
    return new NextResponse(null, { status: 204 });
  } catch {
    // Non-fatal: frontend resets its own state regardless
    return new NextResponse(null, { status: 204 });
  }
}
