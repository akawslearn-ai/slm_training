import { NextRequest } from "next/server";
import { MAX_PER_WINDOW, WINDOW_SECONDS, rateLimit } from "@/lib/ratelimit";

export const runtime = "nodejs";
/* The upstream generation can outlive a short function budget on a cold
   container. Vercel caps this per plan (Hobby is lower); the value here is a
   ceiling, not a guarantee. */
export const maxDuration = 60;

const ENDPOINT = process.env.MODAL_ENDPOINT_URL;
const API_KEY = process.env.MODAL_API_KEY;

export async function POST(req: NextRequest) {
  if (!ENDPOINT || !API_KEY) {
    return Response.json(
      { error: "Inference endpoint is not configured on the server." },
      { status: 503 },
    );
  }

  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0].trim() ??
    req.headers.get("x-real-ip") ??
    "unknown";

  const limit = await rateLimit(ip);
  if (!limit.ok) {
    return Response.json(
      { error: "Too many requests. Wait a minute and try again." },
      {
        status: 429,
        headers: {
          "Retry-After": String(WINDOW_SECONDS),
          "X-RateLimit-Limit": String(MAX_PER_WINDOW),
          "X-RateLimit-Remaining": "0",
        },
      },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Body must be JSON." }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${ENDPOINT}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify(body),
    });
  } catch {
    return Response.json(
      { error: "Could not reach the model server." },
      { status: 502 },
    );
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return Response.json(
      { error: detail.slice(0, 300) || `Model server returned ${upstream.status}.` },
      { status: upstream.status },
    );
  }

  // Pipe the SSE stream straight through; no buffering, so tokens reach the
  // browser as they are produced rather than all at once at the end.
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
