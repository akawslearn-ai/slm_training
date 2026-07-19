"""Generate text from the Phase 5 base model (end-to-end proof it learned)."""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-sample")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

PROMPTS = [
    "The plaintiff shall bear the burden of",
    "The Company's net revenues for the fiscal year",
    "IT IS HEREBY ORDERED that the motion",
    "Item 7. Management's Discussion and Analysis",
    "The court held that the defendant",
]


@app.function(image=image, gpu=config.PRETRAIN_GPU, volumes=VOLUMES, timeout=60 * 15)
def sample(max_new_tokens: int = 60) -> None:
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    model = LlamaForCausalLM.from_pretrained(config.BASE_CKPT_DIR).cuda().eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} params\n")

    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt)], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.convert_tokens_to_ids(
                                     config.SPECIAL_TOKENS["pad_token"]))
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"--- {prompt!r}")
        print(f"{text}\n")


@app.local_entrypoint()
def main(max_new_tokens: int = 60):
    sample.remote(max_new_tokens)
