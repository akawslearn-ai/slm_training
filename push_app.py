"""Phase 6: push the trained base model + tokenizer to the HuggingFace Hub.

Self-contained Modal app (see CLAUDE.md): declares its own image and Volume
handle rather than importing from modal_app.

Runs on CPU. The model artifacts never touch the local machine -- the container
mounts the Volume and uploads straight to the Hub.

  modal run push_app.py --dry-run   # assemble + verify the bundle, no upload
  modal run push_app.py             # assemble, verify, then push
"""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-push")

STAGE_DIR = "/tmp/hf_upload"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==1.26.4",
        "huggingface_hub==0.34.4",
    )
    .add_local_python_source("config")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stage(model_card: str) -> list[str]:
    """Copy weights + tokenizer + card into one flat directory. Returns filenames."""
    import glob
    import os
    import shutil

    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    os.makedirs(STAGE_DIR)

    for src in (config.BASE_CKPT_DIR, config.TOKENIZER_DIR):
        found = glob.glob(f"{src}/*")
        if not found:
            raise RuntimeError(f"nothing to upload in {src}")
        for path in found:
            if os.path.isfile(path):
                shutil.copy2(path, STAGE_DIR)

    with open(f"{STAGE_DIR}/README.md", "w", encoding="utf-8") as fh:
        fh.write(model_card)

    return sorted(os.listdir(STAGE_DIR))


def _verify() -> None:
    """Load the staged bundle exactly as a downstream user would, then generate.

    Catches the failure mode that matters here: weights and tokenizer landing in
    one directory but not actually being loadable as a pair.
    """
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(STAGE_DIR)
    model = LlamaForCausalLM.from_pretrained(STAGE_DIR).eval()

    n_par = sum(p.numel() for p in model.parameters())
    if model.config.vocab_size != config.MODEL.vocab_size:
        raise RuntimeError(f"vocab mismatch: {model.config.vocab_size} "
                           f"!= {config.MODEL.vocab_size}")

    prompt = "The plaintiff shall bear the burden of"
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.convert_tokens_to_ids(
                                 config.SPECIAL_TOKENS["pad_token"]))
    text = tok.decode(out[0].tolist(), skip_special_tokens=True)

    print(f"verify: {n_par:,} params | vocab {model.config.vocab_size} | "
          f"tokenizer {len(tok)} tokens", flush=True)
    print(f"verify: {text!r}", flush=True)


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=8192,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)],
              timeout=60 * 30)
def push(model_card: str, repo_id: str, private: bool, dry_run: bool) -> str:
    import os

    from huggingface_hub import HfApi

    files = _stage(model_card)
    print(f"staged {len(files)} files: {', '.join(files)}", flush=True)
    _verify()

    if dry_run:
        print("dry run: nothing uploaded", flush=True)
        return ""

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=STAGE_DIR,
                      commit_message="Add 125M legal/financial base model")

    url = f"https://huggingface.co/{repo_id}"
    print(f"pushed -> {url}", flush=True)
    return url


@app.local_entrypoint()
def main(repo_id: str = config.HF_REPO, private: bool = False,
         dry_run: bool = False, card: str = "MODEL_CARD.md"):
    with open(card, encoding="utf-8") as fh:
        model_card = fh.read()
    push.remote(model_card, repo_id, private, dry_run)
