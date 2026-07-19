# slm-125m web

Technical page for the SLM-125M base model: a live generation playground, the
architecture, the measured training run, and a scaling calculator that
extrapolates cost and perplexity to larger models.

The page itself is static. The playground calls `/api/generate`, a thin proxy
that forwards to a Modal CPU endpoint (`../serve_app.py`) — the model is far too
large to run in a Vercel function, so inference lives on Modal and Vercel hosts
only the UI.

```
browser → /api/generate (Vercel, holds the key) → Modal CPU container (weights)
```

## Run locally

```bash
npm install
cp .env.example .env.local   # then fill in both values
npm run dev                  # http://localhost:3000
npm run build                # production build; also runs the TS check
```

Without `.env.local` the page renders fine and the playground returns a 503 with
"Inference endpoint is not configured" — everything else works.

## Deploy to Vercel

The repo root is `slm-training/`, so the project root must be set to this
subdirectory or Vercel will not find `package.json`.

```bash
npx vercel        # from web/, first run links + deploys a preview
npx vercel --prod
```

Or via the dashboard: import the Git repo and set **Root Directory** to `web`.
Framework preset, build command, and output directory are all detected.

Then set both environment variables in **Settings → Environment Variables**
(see `.env.example`). They are server-only — no `NEXT_PUBLIC_` prefix — so the
API key is never shipped to the browser:

| Variable | Value |
|---|---|
| `MODAL_ENDPOINT_URL` | printed by `modal deploy serve_app.py`, no trailing slash |
| `MODAL_API_KEY` | must match `SLM_API_KEY` in the Modal secret `slm-api-key` |

## The inference endpoint

Deployed separately from the repo root:

```bash
modal deploy ../serve_app.py    # persistent URL
modal serve  ../serve_app.py    # ephemeral, hot-reloads, for development
```

It runs on CPU with `scaledown_window=300`, so it costs nothing at rest and
cold-starts in ~15-30s after a quiet spell. The playground surfaces that wait
explicitly rather than hiding it behind a spinner. Throughput is ~6-7 tok/s;
moving to a GPU would roughly 10x it at ~$1/hr while warm.

Rate limiting in `app/api/generate/route.ts` is in-memory and therefore
per-instance — a speed bump, not a real control. Move it to Vercel KV or Upstash
before pointing real traffic at this.

## Where the numbers come from

`lib/run-data.ts` is generated from the real training artifacts in the repo root
(`metrics.jsonl`, `index.json`, `eval.json`) — do not hand-edit it. Regenerate it
if the model is retrained.

`lib/scaling.ts` holds the extrapolation model. Its assumptions, and the points
where they stop being trustworthy, are documented in the file header and surfaced
in the "methodology" section of the page itself. In short:

- Loss follows the Chinchilla parametric form with published exponents, but the
  irreducible term `E` is re-solved (0.6614, not 1.69) so the curve passes through
  our own measured point. Perplexity is only comparable within a fixed tokenizer.
- Compute is `6ND`; post-training stages change only the leading coefficient
  (LoRA 4N, DPO 16N, PPO 36N).
- Repeated epochs are discounted per Muennighoff et al. (2023), `R* ≈ 15`.
- MFU is measured only at 125M (22.5% end-to-end). Everything above it is assumed.

## Charts

Palette is the validated dark-mode categorical order (blue / green / magenta),
checked against surface `#141821` — passes lightness band, chroma floor, CVD
separation all-pairs, normal-vision floor, and 3:1 contrast. **Do not reorder the
series colors**: the ordering is the colorblind-safety mechanism, not decoration.
Re-run the validator if you change them.

The loss chart ships a crosshair/tooltip and a table view; the calculator uses two
single-axis small multiples rather than one dual-axis chart.
