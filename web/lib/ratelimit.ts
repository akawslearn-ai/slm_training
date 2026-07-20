/**
 * Rate limiting with two backends.
 *
 * If UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are set, the counter
 * lives in Redis and is therefore shared across every serverless instance --
 * this is the one that actually holds under real traffic.
 *
 * If they are not set, it degrades to a per-instance in-memory map. That stops
 * a single tab hammering the endpoint and nothing more: serverless spins up
 * many instances and each keeps its own counter, so the effective limit is
 * (LIMIT x instances). Fine for a demo, not a control.
 *
 * Deliberately no SDK. Upstash exposes a REST API, so a fixed-window counter is
 * one fetch with INCR + EXPIRE -- adding @upstash/ratelimit would pull in
 * dependencies to save about ten lines.
 */

export const WINDOW_SECONDS = 60;
export const MAX_PER_WINDOW = 12;

export type LimitResult = {
  ok: boolean;
  remaining: number;
  backend: "redis" | "memory";
};

// ---------------------------------------------------------------------------
// in-memory fallback
// ---------------------------------------------------------------------------

const memory = new Map<string, number[]>();

function memoryLimit(ip: string): LimitResult {
  const now = Date.now();
  const windowMs = WINDOW_SECONDS * 1000;
  const recent = (memory.get(ip) ?? []).filter((t) => now - t < windowMs);
  recent.push(now);
  memory.set(ip, recent);

  // Crude unbounded-growth guard. A real eviction policy is not worth it here
  // because this path is the fallback, not the intended backend.
  if (memory.size > 5_000) memory.clear();

  return {
    ok: recent.length <= MAX_PER_WINDOW,
    remaining: Math.max(0, MAX_PER_WINDOW - recent.length),
    backend: "memory",
  };
}

// ---------------------------------------------------------------------------
// redis
// ---------------------------------------------------------------------------

async function redisLimit(ip: string, url: string, token: string): Promise<LimitResult> {
  // Fixed window: the key name carries the window index, so it self-expires
  // and needs no cleanup. INCR is atomic, which is the property that matters.
  const window = Math.floor(Date.now() / (WINDOW_SECONDS * 1000));
  const key = `rl:${ip}:${window}`;

  const res = await fetch(`${url}/pipeline`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify([
      ["INCR", key],
      ["EXPIRE", key, WINDOW_SECONDS],
    ]),
    // Never let the limiter itself become the thing that hangs a request.
    signal: AbortSignal.timeout(1_500),
  });

  if (!res.ok) throw new Error(`upstash ${res.status}`);

  const body = (await res.json()) as { result: number }[];
  const count = Number(body?.[0]?.result ?? 0);
  if (!Number.isFinite(count) || count <= 0) throw new Error("bad upstash reply");

  return {
    ok: count <= MAX_PER_WINDOW,
    remaining: Math.max(0, MAX_PER_WINDOW - count),
    backend: "redis",
  };
}

// ---------------------------------------------------------------------------

/**
 * Resolve Redis REST credentials under either naming convention.
 *
 * Provisioning the same Upstash database two different ways injects two
 * different pairs of variable names:
 *   - Upstash directly, or via the Vercel Marketplace -> UPSTASH_REDIS_REST_*
 *   - Vercel Redis (formerly Vercel KV)               -> KV_REST_API_*
 *
 * Accepting both means the limiter starts working the moment a store is
 * attached, regardless of which route was taken, with no code change.
 */
function redisCredentials(): { url: string; token: string } | null {
  const url = process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;
  return url && token ? { url: url.replace(/\/$/, ""), token } : null;
}

export async function rateLimit(ip: string): Promise<LimitResult> {
  const creds = redisCredentials();

  if (creds) {
    const { url, token } = creds;
    try {
      return await redisLimit(ip, url, token);
    } catch (err) {
      // Fail open to the in-memory limiter rather than 500-ing the request.
      // A rate limiter outage should degrade the control, not the service.
      console.error("rate limit: redis unavailable, using in-memory", err);
    }
  }

  return memoryLimit(ip);
}
