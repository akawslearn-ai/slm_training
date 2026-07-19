---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - llama
  - pretrained
  - small-language-model
datasets:
  - HFforLegal/case-law
  - PleIAs/SEC
  - HuggingFaceFW/fineweb-edu
language:
  - en
---

# slm-125m-legal-base

A 125M-parameter Llama-architecture base model pretrained from scratch on a legal- and
finance-heavy corpus (US case law, SEC filings, and a slice of FineWeb-Edu).

This is a **base model**: it does next-token prediction only. It has not been
instruction-tuned, chat-tuned, or RLHF'd, and it will not follow instructions.

## Model details

| | |
|---|---|
| Parameters | 125,848,320 (tied embeddings) |
| Architecture | Llama (`LlamaForCausalLM`) |
| Layers / hidden / heads | 12 / 768 / 12 (head dim 64, MHA) |
| Intermediate size | 3072 (SwiGLU) |
| Context length | 1024 |
| Vocab | 16,384 (byte-level BPE, trained on this corpus) |
| Position encoding | RoPE (theta 10,000) |
| Norm | RMSNorm (eps 1e-5) |
| Precision | bf16 autocast, fp32 master weights |

The tokenizer is custom and trained on this corpus. A 16K vocab is small by modern
standards; it was chosen to keep the embedding table from dominating a 125M budget.

## Training data

Pretraining used 2,039,072,768 tokens across 1,991,282 packed 1024-token windows,
with a 1% held-out validation split (20,601,856 tokens / 20,119 windows).

| Source | Share of tokens |
|---|---|
| [HFforLegal/case-law](https://huggingface.co/datasets/HFforLegal/case-law) (US) | 35.1% |
| [PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC) | 42.2% |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) | 22.8% |

The mix is legal-first rather than the more common 70/20/10 web-heavy split: the two
legal sources cap at roughly 2B tokens, so the pipeline takes all of both and caps web
at 0.5B.

Data preparation: a 6-step deterministic cleaning chain (line filtering, non-alphanumeric
ratio, minimum document length, repetition/n-gram filtering, language detection, and an
OCR-quality gate applied to case law only), then MinHash near-duplicate removal, then
13-gram decontamination against LexGLUE / CaseHOLD evaluation sets (480,908 eval 13-grams).

## Training procedure

| | |
|---|---|
| Tokens seen | 4.08B (2 epochs) |
| Steps | 7,778 |
| Global batch | 524,288 tokens |
| Optimizer | AdamW (beta 0.9/0.95, weight decay 0.1 on 2-D params only) |
| LR schedule | linear warmup over 200M tokens to 6e-4, cosine decay to 6e-5 |
| Grad clip | 1.0 |
| Hardware | 8x H100 |
| Throughput | ~411k tokens/s (~36% MFU) with `torch.compile` |
| Compute cost | ~$11.40 |

## Evaluation

Perplexity over the **entire** held-out validation split (20,558,208 target tokens):

| Metric | Value |
|---|---|
| Validation loss | 2.2143 |
| Validation perplexity | **9.155** |

This is in-domain perplexity on data drawn from the same mix as training. It is not a
capability benchmark, and it is not comparable to perplexity numbers computed against a
different corpus or with a different tokenizer — a 16K vocab makes perplexity look lower
than it would with a 32K or 50K vocab.

No downstream task benchmarks (MMLU, LegalBench, CaseHOLD, etc.) have been run.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("abhishekai/slm-125m-legal-base")
model = AutoModelForCausalLM.from_pretrained("abhishekai/slm-125m-legal-base")

ids = tok("The plaintiff shall bear the burden of", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=60, do_sample=True,
                     temperature=0.8, top_p=0.95,
                     pad_token_id=tok.convert_tokens_to_ids("<|pad|>"))
print(tok.decode(out[0], skip_special_tokens=True))
```

## Limitations and intended use

This model is a **research and educational artifact** — a demonstration that a coherent
domain-specific base model can be pretrained end-to-end on a small budget. It is not
suitable for production use, and specifically:

- **It does not do arithmetic.** It will confidently produce a revenue figure, a
  comparison figure, and a percentage change that do not follow from each other. This is
  expected at 125M parameters. Do not treat any number it generates as a computation.
- **It is not a source of legal or financial fact.** It reproduces the *register* of
  judicial opinions and SEC filings fluently, which makes fabricated citations, case
  names, docket numbers, and financial figures look plausible. It hallucinates freely.
- **It is not legal or financial advice**, and must not be used as a substitute for a
  qualified professional, nor deployed in any advisory, compliance, or decision-making
  capacity.
- **It does not follow instructions.** It is a base model; prompt it as a text
  continuation model.
- **1024-token context**, so it cannot process a full filing or opinion.
- **It inherits the biases of its sources.** US case law spans a long historical range
  and encodes the prejudices of the courts and eras that produced it; SEC filings are
  corporate self-representation. The model reflects both.
- **English only**, US-jurisdiction-skewed.

## License and attribution

The model weights are released under **Apache-2.0**.

The training corpus draws on three sources with their own terms. None carries a
share-alike or non-commercial clause, so nothing upstream constrains the license of
these weights, but two require attribution:

| Source | License | Obligation |
|---|---|---|
| [HFforLegal/case-law](https://huggingface.co/datasets/HFforLegal/case-law) | CC-BY-4.0 | attribution |
| [PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC) | CC0-1.0 | none |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 | attribution |

This section is that attribution, and it stands whether or not those terms legally
reach model weights — a question that is genuinely unsettled and largely untested in
court. The common position is that weights are not a derivative work of the training
data; Apache-2.0 plus visible credit is the choice that holds up under either answer.

None of this is legal advice. If you intend to use this model commercially, get your
own counsel to look at it.

## Training pipeline

The full pipeline — streaming, cleaning, deduplication, decontamination, tokenizer
training, packing, and distributed pretraining — runs on [Modal](https://modal.com) and
is reproducible. Data preparation (Phases 0-4) cost $1.73 on CPU; pretraining cost
~$11.40 on 8x H100.
