"""Phase 5: pretrain the 125M SLM on the Phase 4 token corpus.

Self-contained Modal app (see CLAUDE.md): declares its own image and Volume
handle rather than importing from modal_app.

  modal run pretrain_app.py::smoke              # cheap 1-GPU validation, ~2 min
  modal run pretrain_app.py::train --epochs 2   # full 8xH100 run
"""

from __future__ import annotations

import modal

import config

app = modal.App(f"{config.PROJECT}-pretrain")

N_GPU = config.PRETRAIN_GPU_COUNT
H100_USD_PER_SEC = 0.001097  # modal.com/pricing, checked 2026-07-19

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==1.26.4",
    )
    .add_local_python_source("config")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _open_shards(directory: str):
    """Memory-map every .bin in a directory as (n_windows, seq_len) uint16."""
    import glob

    import numpy as np

    shards = []
    for path in sorted(glob.glob(f"{directory}/*.bin")):
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        n = arr.shape[0] // config.SEQ_LEN
        if n:
            shards.append(arr[: n * config.SEQ_LEN].reshape(n, config.SEQ_LEN))
    if not shards:
        raise RuntimeError(f"no .bin windows found in {directory}")
    return shards


def _shard_index(shards):
    """Flat (shard_id, row) index over all windows."""
    import numpy as np

    sid = np.concatenate([np.full(s.shape[0], i, dtype=np.int32)
                          for i, s in enumerate(shards)])
    row = np.concatenate([np.arange(s.shape[0], dtype=np.int64) for s in shards])
    return sid, row


def _batches(shards, sid, row, order, batch_size):
    """Yield int64 (batch_size, seq_len) arrays following `order`."""
    import numpy as np

    for start in range(0, len(order) - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        out = np.empty((batch_size, config.SEQ_LEN), dtype=np.int64)
        for j, k in enumerate(idx):
            out[j] = shards[sid[k]][row[k]]
        yield out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _lr_at(tokens_seen: int, total_tokens: int) -> float:
    """Linear warmup on token count, then cosine decay to min_lr."""
    import math

    t = config.TRAIN
    if tokens_seen < t.warmup_tokens:
        return t.lr * tokens_seen / max(1, t.warmup_tokens)
    span = max(1, total_tokens - t.warmup_tokens)
    frac = min(1.0, (tokens_seen - t.warmup_tokens) / span)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1.0 + math.cos(math.pi * frac))


def _worker(rank: int, world_size: int, epochs: float, smoke: bool, budget_usd: float,
            compile_model: bool = True):
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import LlamaConfig, LlamaForCausalLM

    t = config.TRAIN
    torch.manual_seed(t.seed + rank)
    is_main = rank == 0

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_shards = _open_shards(config.TRAIN_TOKENS_DIR)
    val_shards = _open_shards(config.VAL_TOKENS_DIR)
    tr_sid, tr_row = _shard_index(train_shards)
    va_sid, va_row = _shard_index(val_shards)
    n_train = len(tr_sid)

    tokens_per_micro = t.micro_batch_size * config.SEQ_LEN
    accum = max(1, t.global_batch_tokens // (tokens_per_micro * world_size))
    total_tokens = int(n_train * config.SEQ_LEN * epochs)
    total_steps = max(1, total_tokens // t.global_batch_tokens)
    if smoke:
        total_steps = 40

    llama_cfg = LlamaConfig(**config.MODEL.to_llama_kwargs())
    llama_cfg._attn_implementation = "sdpa"  # flash-backed fused attention
    model = LlamaForCausalLM(llama_cfg).to(device)
    model.gradient_checkpointing_disable()
    if is_main:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"model {n_par:,} params | world={world_size} accum={accum} "
              f"steps={total_steps:,} tokens={total_tokens/1e9:.2f}B", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.lr, betas=(t.beta1, t.beta2), fused=True)

    start_step, tokens_seen = 0, 0
    if os.path.exists(config.RESUME_CKPT_PATH) and not smoke:
        ck = torch.load(config.RESUME_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, tokens_seen = ck["step"], ck["tokens_seen"]
        if is_main:
            print(f"resumed from step {start_step:,} ({tokens_seen/1e9:.2f}B tokens)", flush=True)

    # `model` stays the uncompiled module: it owns the same parameters, so
    # state_dict()/save_pretrained() keep clean keys (no _orig_mod. prefix).
    fwd = torch.compile(model) if compile_model else model
    ddp = DDP(fwd, device_ids=[rank]) if world_size > 1 else fwd
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    def evaluate(max_batches: int = 50) -> float:
        model.eval()
        order = np.arange(len(va_sid))[rank::world_size]
        tot, nb = 0.0, 0
        with torch.no_grad():
            for arr in _batches(val_shards, va_sid, va_row, order, t.micro_batch_size):
                ids = torch.from_numpy(arr).to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tot += fwd(input_ids=ids, labels=ids).loss.item()
                nb += 1
                if nb >= max_batches:
                    break
        model.train()
        loss = torch.tensor([tot / max(1, nb)], device=device)
        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return loss.item()

    model.train()
    t0 = time.time()
    step = start_step
    halted = False
    rng = np.random.default_rng(t.seed)
    # Steady-state timers start after warmup/compile so tok/s excludes it.
    t_steady, tok_steady = None, 0

    while step < total_steps and not halted:
        order = rng.permutation(n_train)[rank::world_size]
        stream = _batches(train_shards, tr_sid, tr_row, order, t.micro_batch_size)
        exhausted = False
        while step < total_steps and not exhausted:
            lr = _lr_at(tokens_seen, total_tokens)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            micro_loss = 0.0
            for i in range(accum):
                try:
                    arr = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                ids = torch.from_numpy(arr).to(device, non_blocking=True)
                if world_size > 1:
                    ddp.require_backward_grad_sync = (i == accum - 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ddp(input_ids=ids, labels=ids).loss / accum
                loss.backward()
                micro_loss += loss.item()
            if exhausted:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            opt.step()
            tokens_seen += tokens_per_micro * accum * world_size
            step += 1

            elapsed = time.time() - t0
            spend = elapsed * world_size * H100_USD_PER_SEC
            if t_steady is None and step >= start_step + 3:
                t_steady, tok_steady = time.time(), tokens_seen

            if is_main and step % t.log_every_steps == 0:
                tps = ((tokens_seen - tok_steady) / max(1e-9, time.time() - t_steady)
                       if t_steady else tokens_seen / max(1e-9, elapsed))
                print(f"step {step:>6}/{total_steps} loss {micro_loss:.4f} lr {lr:.2e} "
                      f"tok {tokens_seen/1e9:.2f}B {tps/1e3:.0f}k tok/s ${spend:.2f}",
                      flush=True)
                with open(config.METRICS_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, "loss": micro_loss, "lr": lr,
                                         "tokens": tokens_seen, "usd": round(spend, 4)}) + "\n")

            if step % t.eval_every_steps == 0 or step == total_steps:
                vl = evaluate()
                if is_main:
                    print(f"  [eval] step {step} val_loss {vl:.4f} ppl {np.exp(vl):.2f}",
                          flush=True)

            if (step % t.ckpt_every_steps == 0 or step == total_steps) and is_main and not smoke:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "tokens_seen": tokens_seen},
                           config.RESUME_CKPT_PATH)
                volume.commit()
                print(f"  [ckpt] step {step} -> {config.RESUME_CKPT_PATH}", flush=True)

            if spend > budget_usd:
                if is_main:
                    print(f"BUDGET STOP: ${spend:.2f} > ${budget_usd:.2f} at step {step}",
                          flush=True)
                halted = True
                break

    final_val = evaluate(max_batches=200)
    if is_main:
        elapsed = time.time() - t0
        spend = elapsed * world_size * H100_USD_PER_SEC
        if not smoke:
            model.save_pretrained(config.BASE_CKPT_DIR)
            volume.commit()
        print(f"DONE step={step} val_loss={final_val:.4f} ppl={np.exp(final_val):.2f} "
              f"wall={elapsed/60:.1f}min cost=${spend:.2f}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


def _launch(world_size: int, epochs: float, smoke: bool, budget_usd: float,
            compile_model: bool):
    import os

    import torch.multiprocessing as mp

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    args = (world_size, epochs, smoke, budget_usd, compile_model)
    if world_size == 1:
        _worker(0, *args)
    else:
        mp.spawn(_worker, args=args, nprocs=world_size)


@app.function(image=gpu_image, gpu=f"{config.PRETRAIN_GPU}:{N_GPU}",
              volumes=VOLUMES, timeout=60 * 60 * 4)
def pretrain(epochs: float = 2.0, budget_usd: float = config.BUDGET_CAP_USD,
             compile_model: bool = True) -> None:
    _launch(N_GPU, epochs, False, budget_usd, compile_model)


@app.function(image=gpu_image, gpu=config.PRETRAIN_GPU, volumes=VOLUMES,
              timeout=60 * 60 * 6)
def pretrain_1gpu(epochs: float = 2.0, budget_usd: float = config.BUDGET_CAP_USD,
                  compile_model: bool = True) -> None:
    """Single-GPU path: same total cost as 8x (cost tracks FLOPs), ~8x wall-clock.
    accum becomes 16, so the 524,288-token global batch is unchanged."""
    _launch(1, epochs, False, budget_usd, compile_model)


@app.function(image=gpu_image, gpu=config.PRETRAIN_GPU, volumes=VOLUMES, timeout=60 * 30)
def pretrain_smoke(compile_model: bool = True) -> None:
    _launch(1, 0.01, True, config.BUDGET_CAP_USD, compile_model)


@app.local_entrypoint()
def smoke(compile_model: bool = True):
    pretrain_smoke.remote(compile_model)


@app.local_entrypoint()
def train(epochs: float = 2.0, budget_usd: float = config.BUDGET_CAP_USD,
          compile_model: bool = True):
    pretrain.remote(epochs, budget_usd, compile_model)


@app.local_entrypoint()
def train1(epochs: float = 2.0, budget_usd: float = config.BUDGET_CAP_USD,
           compile_model: bool = True):
    pretrain_1gpu.remote(epochs, budget_usd, compile_model)
