"""Standalone Phase 4 verification (does not modify the four canonical files)."""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-verify")

# Same spec as modal_app.ml_image so the build cache is reused.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
    .pip_install("transformers==4.46.3")
    .add_local_python_source("config", "cleaning", "dedup")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


@app.function(image=image, volumes=VOLUMES, timeout=60 * 15)
def verify() -> dict:
    import glob
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    with open(f"{config.TOKENS_DIR}/index.json", encoding="utf-8") as fh:
        idx = json.load(fh)

    itemsize = np.dtype(config.TOKENS_DTYPE).itemsize
    bad = []
    tot_train_bytes = tot_val_bytes = 0
    for s in idx["shards"]:
        stem = f"{s['source']}-{s['shard']:03d}.bin"
        for split, key in (("train", "train_tokens"), ("val", "val_tokens")):
            path = f"{config.TOKENS_DIR}/{split}/{stem}"
            size = os.path.getsize(path)
            expect = s[key] * itemsize
            if split == "train":
                tot_train_bytes += size
            else:
                tot_val_bytes += size
            if size != expect:
                bad.append((path, size, expect))

    print(f"train bytes {tot_train_bytes:,} ({tot_train_bytes/1e9:.2f} GB)")
    print(f"val   bytes {tot_val_bytes:,} ({tot_val_bytes/1e6:.1f} MB)")
    print(f"expected train bytes {idx['train_tokens']*itemsize:,}")
    print(f"SIZE MISMATCHES: {len(bad)}")
    for b in bad:
        print(f"  MISMATCH {b}")

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])

    for src in ("case-law", "sec", "fineweb-edu"):
        path = sorted(glob.glob(f"{config.TRAIN_TOKENS_DIR}/{src}-*.bin"))[0]
        arr = np.fromfile(path, dtype=config.TOKENS_DTYPE, count=config.SEQ_LEN)
        big = np.fromfile(path, dtype=config.TOKENS_DTYPE, count=200 * config.SEQ_LEN)
        print(f"\n[{src}] {os.path.basename(path)} min={arr.min()} max={arr.max()} "
              f"in_range={bool(big.max() < config.MODEL.vocab_size)} "
              f"eos_in_200_win={int((big == eos_id).sum())}")
        print(f"  decoded: {tok.decode(arr[:50].tolist())[:220]!r}")

    return {"mismatches": len(bad), "train_bytes": tot_train_bytes,
            "val_bytes": tot_val_bytes}


@app.local_entrypoint()
def main():
    verify.remote()
