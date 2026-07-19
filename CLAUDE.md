# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Modal-based CPU data pipeline that builds pretraining data for a 125M-parameter legal/financial
SLM: stream → clean → dedup/decontaminate → train tokenizer → tokenize/pack. Output is a packed
uint16 token corpus on a Modal Volume. `replication_guide.md` is the authoritative spec; it is the
brief the four source files were written from.

Pretraining (Phase 5) and HF model push (Phase 6) are **not** implemented here.

## Commands

All Modal runs need `PYTHONIOENCODING=utf-8` on Windows (see Environment below).

```bash
.venv/Scripts/python config.py              # local sanity check, no Modal
                                            # expect: model: 125,847,552 params (~125.8M)

modal run modal_app.py::main                # Phase 0a smoke test (10 docs/source)
modal run modal_app.py::measure             # Phase 0b per-source token yield projection
modal run modal_app.py::clean --fineweb-shards 5   # Phase 1 (20 workers)
modal run modal_app.py::clean --only case-law      # redo one source
modal run modal_app.py::dedup               # Phase 2 (add --no-compute-sigs to reuse MinHash sigs)
modal run modal_app.py::tokenizer           # Phase 3
modal run modal_app.py::tokenize            # Phase 4 (14 workers)
modal run modal_app.py::ocr                 # optional: OCR threshold histogram
modal run verify_app.py                     # integrity check of Phase 4 output
```

There is no test suite. `verify_app.py` is the verification path: it checks every `.bin` byte size
against `index.json`, asserts token IDs are under `vocab_size`, and decodes real windows back to text.

## Architecture

**`config.py` is the single source of truth.** Every other module imports it. Data mix, budgets,
thresholds, paths, and model shape live there and nowhere else. Change data volume by editing
`Source.token_budget`, then re-run Phase 1 and everything downstream in order.

**Phases are sequential and stateful on the Volume.** Each phase reads the previous phase's output
from `/data` and commits its own. Skipping a phase or running one against stale output silently
produces wrong results rather than erroring.

```
/data/clean/<source>/shard-XX.txt    Phase 1  cleaned, one doc per line
/data/corpus/<source>/shard-XX.txt   Phase 2  deduped + decontaminated
/data/tokenizer/                     Phase 3  16K byte-level BPE
/data/tokens/{train,val}/*.bin       Phase 4  packed uint16 1024-token windows
/data/tokens/index.json              Phase 4  authoritative token/window counts
```

**Everything heavy is fanned out one worker per shard** (Phases 1, 2, 4). This is deliberate: Modal
can preempt a long single container and restart it from zero. Do not consolidate into one big
container.

**Pure logic is separated from Modal orchestration.** `cleaning.py` (6-step deterministic chain) and
`dedup.py` (hashing/shingling helpers) are pure functions with no Modal imports, so they can be
tested and reasoned about locally. `modal_app.py` holds all app/image/volume/function definitions.

**The data mix is legal-first, NOT 70/20/10.** The two legal sources cap at ~2B tokens total, so the
pipeline takes all of case-law and SEC and caps web at 0.5B. Realized ~35/42/23. Do not "fix" this
toward 70/20/10 — it is impossible with these datasets.

## Non-obvious constraints

**Modal image ordering.** All `pip_install`/`apt_install` must precede `add_local_python_source`, or
the build errors.

**`add_local_python_source("config", "cleaning", "dedup")` is the complete set of bundled modules.**
A new Modal file that does `from modal_app import ...` will crash-loop every container with
`ModuleNotFoundError`. Make new Modal entrypoints self-contained: redeclare the image with an
identical spec (the build cache is reused) and its own `Volume.from_name`.

**Hardcoded shard counts must match reality.** `CLEAN_SHARDS` (Phase 2) and `TOKENIZE_SHARDS`
(Phase 4) are literals in `modal_app.py`. If the HuggingFace parquet listing returns a different
shard count than Phase 1 produced, Phase 2 reads missing files. Verify Phase 1's per-source shard
count before running dedup.

**Token counts before Phase 4 are a chars/4 proxy, not real counts.** Only `index.json` has true
tokenizer counts. Measured real/proxy ratios: case-law 0.896, sec 0.796, fineweb-edu 0.938.

**`casehold/casehold` parquet does not resolve.** Phase 2 logs `could not load casehold/casehold`
once per worker — expected, not a failure. The LexGLUE `case_hold` config covers the same benchmark;
confirm `480,908 eval 13-grams loaded` appears. Each worker rebuilds the contamination set
independently (20 redundant downloads).

**The OCR gate needs `/usr/share/dict/words`**, provided by the `wamerican` apt package. It only
runs for sources with `strict_ocr=True` (case-law only).

**`replication_guide.md` line ~641 is corrupted.** The parquet API URL is wrapped in markdown link
syntax, which is a Python syntax error. `modal_app.py` has the correct plain form. The guide also
says Phase 1 launches 16 workers; it launches 20 (10+5+5).

## Environment (Windows)

- `export PYTHONIOENCODING=utf-8` before any `modal run` — the console's cp1252 codepage crashes on
  the checkmark glyph Modal prints on success.
- `export MSYS_NO_PATHCONV=1` for `modal volume` commands with absolute paths. Git Bash rewrites
  `/tokens` into a Windows path, and Modal reports a misleading "No such file or directory".
  `modal volume ls slm-125m` (no path) always works.
- Do not pipe long background `modal run` through `tail` — it buffers until EOF, so you get no
  incremental output.
- Auth resolves from `~/.modal.toml`; `.env.local` is optional (template in `.env.local.example`).

## Realized baseline (2026-07-18 run)

train 2,039,072,768 tokens / 1,991,282 windows; val 20,601,856 / 20,119 (1.0002%).
Mix: case-law 35.1%, sec 42.2%, fineweb-edu 22.8%. Cost well under $1, all CPU.

The guide predicts 2.19B with case-law at 863M. This run measured case-law at 722M. SEC and
fineweb-edu reproduce the guide to within 0.1%; the difference is case-law tokenizer compression
(4.47 vs the guide's implied 3.74 chars/token), not data loss — Phase 1/2 proxy totals matched the
guide exactly and `verify_app.py` reports zero byte mismatches.
