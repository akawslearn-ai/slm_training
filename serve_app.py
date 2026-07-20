"""Phase 7: streaming inference endpoint for the web playground.

Self-contained Modal app (see CLAUDE.md): declares its own image and Volume
handle rather than importing from modal_app.

Runs on CPU. A 125M model decodes at roughly reading speed on 8 cores, and
scale-to-zero means an idle endpoint costs nothing. The trade is a ~20-30s cold
start on the first request after idle -- the web UI surfaces that honestly
rather than hiding it behind a spinner.

  modal serve serve_app.py    # ephemeral, hot-reloads on save, for development
  modal deploy serve_app.py   # persistent URL

The deployed URL is printed on deploy. Point MODAL_ENDPOINT_URL at it.
"""

# NOTE: deliberately no `from __future__ import annotations` here, unlike the
# other Modal apps in this repo. PEP 563 turns annotations into strings, and
# FastAPI then cannot resolve the request model below -- it is defined inside
# web(), so it is not in module globals, and the body silently degrades into a
# query parameter (422 "Field required"). Keep annotations eager in this file.

import modal

import config

app = modal.App(f"{config.PROJECT}-serve")

MAX_NEW_TOKENS = 200
MAX_PROMPT_CHARS = 2_000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==1.26.4",
        "fastapi[standard]==0.115.6",
    )
    .add_local_python_source("config")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

API_KEY_SECRET = "slm-api-key"


@app.cls(
    image=image,
    volumes=VOLUMES,
    cpu=8,
    memory=8192,
    secrets=[modal.Secret.from_name(API_KEY_SECRET)],
    # Stay warm for 5 min after the last request, then scale to zero. A visitor
    # who arrives during a session never pays the cold start twice.
    scaledown_window=300,
    timeout=60 * 10,
    # HARD COST CEILING. Without this, Modal autoscales on concurrent load and a
    # burst of traffic -- accidental or hostile -- fans out into arbitrarily many
    # billed containers. This is the protection that actually bounds spend: a
    # per-IP rate limiter does nothing against distributed requests, and caps the
    # front door while leaving the backend unbounded.
    #
    # Two containers is enough for a demo (one visitor does not block another).
    # Excess concurrency queues instead of scaling, so the failure mode under
    # abuse is "slow", not "expensive".
    max_containers=2,
)
class Server:
    @modal.enter()
    def load(self):
        """Runs once per container, not per request. This is the whole reason
        the playground is usable -- loading 503MB of weights per request would
        put every generation behind a 20s stall."""
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM

        torch.set_num_threads(8)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.model = LlamaForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.float32
        ).eval()
        self.pad_id = self.tokenizer.convert_tokens_to_ids(
            config.SPECIAL_TOKENS["pad_token"]
        )
        print(f"loaded {sum(p.numel() for p in self.model.parameters()):,} params", flush=True)

    @modal.asgi_app()
    def web(self):
        import json
        import os
        import threading

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        api = FastAPI()
        expected_key = os.environ["SLM_API_KEY"]

        def clamp(v, lo, hi, default):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, v))

        @api.get("/health")
        def health():
            return {"ok": True, "model": config.HF_REPO, "revision": 2}

        # The body is parsed by hand rather than via a Pydantic parameter
        # annotation. FastAPI resolves annotations through the *module* globals,
        # and a request model declared inside this method is not there -- it
        # degrades into a query parameter and every POST 422s. Parsing the raw
        # Request sidesteps that entirely and lets auth run before validation,
        # so a missing key gets a 401 instead of a confusing 422.
        @api.post("/generate")
        async def generate(request: Request):
            import hmac

            key = request.headers.get("x-api-key")
            if not key or not hmac.compare_digest(key, expected_key):
                raise HTTPException(status_code=401, detail="invalid api key")

            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")

            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise HTTPException(status_code=400, detail="prompt is required")
            if len(prompt) > MAX_PROMPT_CHARS:
                raise HTTPException(
                    status_code=400, detail=f"prompt exceeds {MAX_PROMPT_CHARS} chars"
                )

            max_new = int(clamp(body.get("max_new_tokens", 90), 1, MAX_NEW_TOKENS, 90))
            temperature = clamp(body.get("temperature", 0.8), 0.0, 2.0, 0.8)
            top_p = clamp(body.get("top_p", 0.95), 0.01, 1.0, 0.95)
            top_k = int(clamp(body.get("top_k", 50), 0, 200, 50))

            torch = self.torch
            input_ids = torch.tensor([self.tokenizer.encode(prompt)])
            if input_ids.shape[1] >= config.MODEL.max_position_embeddings:
                raise HTTPException(status_code=400, detail="prompt exceeds context window")

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_new,
                streamer=streamer,
                pad_token_id=self.pad_id,
                # temperature 0 means greedy; HF errors if do_sample is on with temp 0
                do_sample=temperature > 0,
            )
            if temperature > 0:
                kwargs.update(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                )

            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()

            def sse():
                n = 0
                try:
                    for chunk in streamer:
                        if chunk:
                            n += 1
                            yield f"data: {json.dumps({'token': chunk})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'chunks': n})}\n\n"
                except Exception as exc:  # surface failures to the client, don't hang
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                finally:
                    thread.join(timeout=5)

            return StreamingResponse(
                sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return api
