import { NextRequest, NextResponse } from "next/server";

/**
 * GET /images/[itemId]
 * Proxies the image request to the Python backend so the browser can load
 * images from the same origin (no CORS issues, no hardcoded backend URLs).
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ itemId: string }> }
) {
  const { itemId } = await params;
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  try {
    const upstream = await fetch(`${backendUrl}/images/${itemId}`, {
      // Pass cache headers through
      next: { revalidate: 86400 },
    });

    if (!upstream.ok) {
      return new NextResponse(null, { status: upstream.status });
    }

    const imageBuffer = await upstream.arrayBuffer();
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") ?? "image/jpeg",
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch {
    return new NextResponse(null, { status: 502 });
  }
}
