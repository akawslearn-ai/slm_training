/**
 * Scaling extrapolation, calibrated to the slm-125m run.
 *
 * Loss model is the Chinchilla parametric form (Hoffmann et al., 2022):
 *
 *     L(N, D) = E + A/N^alpha + B/D^beta
 *
 * The exponents and coefficients are Chinchilla's published fit. The irreducible
 * term E is NOT Chinchilla's 1.69 -- it is re-solved so the curve passes exactly
 * through our own measured point (125.8M params, 4.08B tokens, val loss 2.2143).
 *
 * Why re-solve E: perplexity is only comparable within a fixed tokenizer and
 * corpus. This model uses a 16K-vocab BPE on a legal/financial corpus, which is
 * far more predictable than the general web text Chinchilla fit against, so its
 * entropy floor is genuinely lower. Keeping Chinchilla's E would put the curve
 * ~1.5 nats above a number we actually measured.
 *
 * What this means for the output: the SHAPE of the curve is borrowed and the
 * ANCHOR is ours. One anchor point cannot constrain two exponents, so the
 * further you extrapolate from 125M, the more this leans on the assumption that
 * our corpus scales like Chinchilla's. Treat 10x as indicative and 100x as a
 * back-of-envelope.
 */

export const LOSS_MODEL = {
  A: 406.4,
  alpha: 0.34,
  B: 410.7,
  beta: 0.28,
  /** Solved from our measured point, not Chinchilla's 1.69. */
  E: 0.6614,
} as const;

/** Chinchilla compute-optimal ratio: ~20 tokens per parameter. */
export const CHINCHILLA_RATIO = 20;

/**
 * Diminishing returns from repeating data (Muennighoff et al., 2023,
 * "Scaling Data-Constrained Language Models"). Repeated tokens are worth less
 * than fresh ones; the decay constant R* ~= 15 epochs.
 *
 * Up to ~4 epochs, repeated data is nearly as good as new data. Past ~15 it is
 * close to worthless -- you still pay full compute for every token.
 */
export const REPEAT_DECAY = 15;

export function effectiveTokens(uniqueTokens: number, epochs: number): number {
  if (epochs <= 1) return uniqueTokens * epochs;
  const extra = epochs - 1;
  return uniqueTokens * (1 + REPEAT_DECAY * (1 - Math.exp(-extra / REPEAT_DECAY)));
}

/** Predicted validation loss (nats/token) for N params and D effective tokens. */
export function predictLoss(params: number, effTokens: number): number {
  const { A, alpha, B, beta, E } = LOSS_MODEL;
  return E + A / Math.pow(params, alpha) + B / Math.pow(effTokens, beta);
}

export function perplexity(loss: number): number {
  return Math.exp(loss);
}

// ---------------------------------------------------------------------------
// hardware
// ---------------------------------------------------------------------------

export type Gpu = {
  id: string;
  name: string;
  /** Dense bf16 peak, FLOP/s. */
  peakFlops: number;
  /** USD per GPU-hour (Modal on-demand). */
  usdPerHour: number;
  memoryGb: number;
};

export const GPUS: Gpu[] = [
  { id: "h100", name: "H100 80GB", peakFlops: 989.5e12, usdPerHour: 3.95, memoryGb: 80 },
  { id: "a100", name: "A100 80GB", peakFlops: 312e12, usdPerHour: 2.5, memoryGb: 80 },
];

/**
 * Model FLOPs Utilisation. Our 125M run measured 22.5% end-to-end (31-36% in
 * steady state -- the gap is torch.compile and CUDA init, which a short run
 * cannot amortise).
 *
 * Small models are latency-bound on kernel launches; larger models fill the
 * pipes better. This curve reflects commonly reported training MFU by scale.
 */
/*
 * Thresholds deliberately sit BETWEEN presets, not on them. A boundary at
 * exactly 1e9 would put the "1B" preset (1.1e9 params) in the large-model
 * bucket and quietly hand it a more optimistic efficiency assumption than its
 * label implies.
 */
export function defaultMfu(params: number): number {
  if (params <= 200e6) return 0.3;
  if (params <= 1.5e9) return 0.36;
  if (params <= 10e9) return 0.42;
  if (params <= 35e9) return 0.46;
  return 0.48;
}

// ---------------------------------------------------------------------------
// training stages
// ---------------------------------------------------------------------------

/**
 * FLOPs per parameter per token, by stage.
 *
 * Pretraining and full fine-tuning are the standard 6N (2N forward, 4N backward).
 * LoRA skips weight gradients for the frozen base, leaving ~4N.
 * DPO processes a chosen and a rejected completion, and runs a frozen reference
 * model forward alongside the policy: 2 x (6N policy + 2N reference) = 16N.
 * PPO additionally runs reward and value models plus autoregressive rollouts;
 * ~6x full SFT is the common empirical range.
 */
export const STAGE_FLOPS_PER_PARAM_TOKEN = {
  pretrain: 6,
  sftFull: 6,
  sftLora: 4,
  dpo: 16,
  ppo: 36,
  inference: 2,
} as const;

export type StageId = keyof typeof STAGE_FLOPS_PER_PARAM_TOKEN;

export type ComputeResult = {
  flops: number;
  gpuHours: number;
  wallHours: number;
  usd: number;
};

export function computeCost(
  params: number,
  tokens: number,
  flopsPerParamToken: number,
  gpu: Gpu,
  gpuCount: number,
  mfu: number,
): ComputeResult {
  const flops = flopsPerParamToken * params * tokens;
  const effectiveFlopsPerSec = gpu.peakFlops * mfu;
  const gpuSeconds = flops / effectiveFlopsPerSec;
  const gpuHours = gpuSeconds / 3600;
  return {
    flops,
    gpuHours,
    wallHours: gpuHours / gpuCount,
    usd: gpuHours * gpu.usdPerHour,
  };
}

// ---------------------------------------------------------------------------
// model presets
// ---------------------------------------------------------------------------

export type Preset = {
  id: string;
  label: string;
  params: number;
  layers: number;
  hidden: number;
  heads: number;
  note?: string;
};

export const PRESETS: Preset[] = [
  { id: "125m", label: "125M", params: 125_848_320, layers: 12, hidden: 768, heads: 12, note: "this model" },
  { id: "300m", label: "300M", params: 302_000_000, layers: 24, hidden: 1024, heads: 16 },
  { id: "1b", label: "1B", params: 1_100_000_000, layers: 16, hidden: 2048, heads: 32 },
  { id: "3b", label: "3B", params: 3_200_000_000, layers: 26, hidden: 2560, heads: 32 },
  { id: "7b", label: "7B", params: 6_700_000_000, layers: 32, hidden: 4096, heads: 32 },
  { id: "13b", label: "13B", params: 13_000_000_000, layers: 40, hidden: 5120, heads: 40 },
  { id: "30b", label: "30B", params: 30_000_000_000, layers: 60, hidden: 6656, heads: 52 },
  { id: "50b", label: "50B", params: 50_000_000_000, layers: 64, hidden: 8192, heads: 64 },
];

// ---------------------------------------------------------------------------
// memory footprint (the constraint that actually stops people)
// ---------------------------------------------------------------------------

/**
 * Bytes per parameter for mixed-precision AdamW training:
 *   bf16 weights 2 + bf16 grads 2 + fp32 master 4 + fp32 m 4 + fp32 v 4 = 16
 * Activations are excluded -- they depend on batch, sequence and checkpointing.
 * This is the floor, not the requirement.
 */
export const BYTES_PER_PARAM_TRAINING = 16;

export function minGpusForMemory(params: number, gpu: Gpu): number {
  const bytes = params * BYTES_PER_PARAM_TRAINING;
  const usableBytes = gpu.memoryGb * 1e9 * 0.8; // 20% headroom for activations
  return Math.max(1, Math.ceil(bytes / usableBytes));
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

export function fmtTokens(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

export function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 10e9 ? 0 : 1)}B`;
  return `${Math.round(n / 1e6)}M`;
}

export function fmtUsd(n: number): string {
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  if (n >= 10) return `$${n.toFixed(0)}`;
  if (n >= 1) return `$${n.toFixed(2)}`;
  return `$${n.toFixed(3)}`;
}

export function fmtHours(h: number): string {
  if (h >= 24 * 365) return `${(h / (24 * 365)).toFixed(1)} yr`;
  if (h >= 24 * 30) return `${(h / (24 * 30)).toFixed(1)} mo`;
  if (h >= 24) return `${(h / 24).toFixed(1)} d`;
  if (h >= 1) return `${h.toFixed(1)} h`;
  return `${(h * 60).toFixed(0)} min`;
}

export function fmtFlops(f: number): string {
  if (f >= 1e24) return `${(f / 1e24).toFixed(2)} YFLOP`;
  if (f >= 1e21) return `${(f / 1e21).toFixed(2)} ZFLOP`;
  if (f >= 1e18) return `${(f / 1e18).toFixed(2)} EFLOP`;
  return `${(f / 1e15).toFixed(2)} PFLOP`;
}
