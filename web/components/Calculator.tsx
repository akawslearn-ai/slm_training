"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  CHINCHILLA_RATIO,
  GPUS,
  PRESETS,
  STAGE_FLOPS_PER_PARAM_TOKEN,
  computeCost,
  defaultMfu,
  effectiveTokens,
  fmtFlops,
  fmtHours,
  fmtParams,
  fmtTokens,
  fmtUsd,
  minGpusForMemory,
  perplexity,
  predictLoss,
} from "@/lib/scaling";
import { RUN } from "@/lib/run-data";

function Control({
  label,
  value,
  children,
  hint,
}: {
  label: string;
  value: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 mb-1.5">
        <label className="text-[12.5px]" style={{ color: "var(--ink-secondary)" }}>
          {label}
        </label>
        <span className="mono text-[12.5px] font-semibold">{value}</span>
      </div>
      {children}
      {hint ? (
        <p className="text-[11px] mt-1" style={{ color: "var(--ink-muted)" }}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** Small single-series chart. Shares the x-domain with its sibling (small multiples). */
function MiniChart({
  points,
  markIndex,
  label,
  format,
  color,
  invertBetter,
}: {
  points: { x: number; y: number }[];
  markIndex: number;
  label: string;
  format: (n: number) => string;
  color: string;
  invertBetter?: boolean;
}) {
  const W = 340;
  const H = 130;
  const P = { t: 14, r: 12, b: 24, l: 44 };

  const ys = points.map((p) => p.y);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const span = yMax - yMin || 1;

  const sx = (i: number) => P.l + (i / (points.length - 1)) * (W - P.l - P.r);
  const sy = (y: number) => P.t + (1 - (y - yMin) / span) * (H - P.t - P.b);

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(i)},${sy(p.y)}`).join(" ");
  const mark = points[markIndex];

  return (
    <div>
      <div className="eyebrow mb-2">{label}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img" aria-label={label}>
        {[yMin, yMin + span / 2, yMax].map((v, i) => (
          <g key={i}>
            <line x1={P.l} x2={W - P.r} y1={sy(v)} y2={sy(v)} stroke="var(--grid)" strokeWidth="1" />
            <text
              x={P.l - 7}
              y={sy(v)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize="9.5"
              fill="var(--ink-muted)"
              className="mono"
            >
              {format(v)}
            </text>
          </g>
        ))}
        <line
          x1={P.l}
          x2={W - P.r}
          y1={H - P.b}
          y2={H - P.b}
          stroke="var(--axis)"
          strokeWidth="1"
        />
        {[0, Math.floor(points.length / 2), points.length - 1].map((i) => (
          <text
            key={i}
            x={sx(i)}
            y={H - P.b + 15}
            textAnchor="middle"
            fontSize="9.5"
            fill="var(--ink-muted)"
            className="mono"
          >
            {points[i].x}
          </text>
        ))}
        <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" />
        <line
          x1={sx(markIndex)}
          x2={sx(markIndex)}
          y1={P.t}
          y2={H - P.b}
          stroke="var(--hairline-strong)"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
        <circle
          cx={sx(markIndex)}
          cy={sy(mark.y)}
          r="4.5"
          fill={color}
          stroke="var(--surface-1)"
          strokeWidth="2"
        />
      </svg>
      <div className="mono text-[11px] mt-1 text-center" style={{ color: "var(--ink-muted)" }}>
        epochs {invertBetter ? "→ lower is better" : ""}
      </div>
    </div>
  );
}

export function Calculator() {
  const [presetId, setPresetId] = useState("1b");
  const [epochs, setEpochs] = useState(2);
  const [gpuId, setGpuId] = useState("h100");
  const [gpuCount, setGpuCount] = useState(64);
  const [mfuOverride, setMfuOverride] = useState<number | null>(null);
  const [dataRatio, setDataRatio] = useState(CHINCHILLA_RATIO);

  const preset = PRESETS.find((p) => p.id === presetId)!;
  const gpu = GPUS.find((g) => g.id === gpuId)!;
  const N = preset.params;
  const mfu = mfuOverride ?? defaultMfu(N);
  const uniqueTokens = N * dataRatio;

  const result = useMemo(() => {
    const nominal = uniqueTokens * epochs;
    const eff = effectiveTokens(uniqueTokens, epochs);
    const loss = predictLoss(N, eff);
    const cost = computeCost(N, nominal, STAGE_FLOPS_PER_PARAM_TOKEN.pretrain, gpu, gpuCount, mfu);
    return { nominal, eff, loss, ppl: perplexity(loss), ...cost };
  }, [N, uniqueTokens, epochs, gpu, gpuCount, mfu]);

  // Sweep epochs 1..12 for the two small multiples.
  const sweep = useMemo(() => {
    const out: { x: number; ppl: number; usd: number }[] = [];
    for (let e = 1; e <= 12; e++) {
      const eff = effectiveTokens(uniqueTokens, e);
      const c = computeCost(
        N,
        uniqueTokens * e,
        STAGE_FLOPS_PER_PARAM_TOKEN.pretrain,
        gpu,
        gpuCount,
        mfu,
      );
      out.push({ x: e, ppl: perplexity(predictLoss(N, eff)), usd: c.usd });
    }
    return out;
  }, [N, uniqueTokens, gpu, gpuCount, mfu]);

  const memFloor = minGpusForMemory(N, gpu);
  const memWarn = gpuCount < memFloor;

  // Marginal value of the next epoch — the number that actually drives decisions.
  const nextEff = effectiveTokens(uniqueTokens, epochs + 1);
  const nextPpl = perplexity(predictLoss(N, nextEff));
  const nextCost = computeCost(
    N,
    uniqueTokens * (epochs + 1),
    STAGE_FLOPS_PER_PARAM_TOKEN.pretrain,
    gpu,
    gpuCount,
    mfu,
  );
  const deltaPpl = result.ppl - nextPpl;
  const deltaUsd = nextCost.usd - result.usd;

  // Downstream stages, sized as fractions of pretraining data.
  const sftTokens = Math.max(5e6, uniqueTokens * 0.005);
  const dpoTokens = Math.max(2e6, uniqueTokens * 0.001);
  const stages = [
    {
      id: "pretrain",
      name: "Pretraining",
      detail: `${fmtTokens(result.nominal)} tokens · ${epochs} epoch${epochs > 1 ? "s" : ""}`,
      ...computeCost(N, result.nominal, STAGE_FLOPS_PER_PARAM_TOKEN.pretrain, gpu, gpuCount, mfu),
    },
    {
      id: "sftFull",
      name: "SFT (full fine-tune)",
      detail: `${fmtTokens(sftTokens * 3)} tokens · 3 epochs · 6N`,
      ...computeCost(N, sftTokens * 3, STAGE_FLOPS_PER_PARAM_TOKEN.sftFull, gpu, gpuCount, mfu),
    },
    {
      id: "sftLora",
      name: "SFT (LoRA)",
      detail: `same data · 4N, no weight grads`,
      ...computeCost(N, sftTokens * 3, STAGE_FLOPS_PER_PARAM_TOKEN.sftLora, gpu, gpuCount, mfu),
    },
    {
      id: "dpo",
      name: "DPO",
      detail: `${fmtTokens(dpoTokens)} pairs · 16N, policy + frozen ref`,
      ...computeCost(N, dpoTokens, STAGE_FLOPS_PER_PARAM_TOKEN.dpo, gpu, gpuCount, mfu),
    },
    {
      id: "ppo",
      name: "PPO / RLHF",
      detail: `${fmtTokens(dpoTokens)} tokens · 36N, 4 models + rollouts`,
      ...computeCost(N, dpoTokens, STAGE_FLOPS_PER_PARAM_TOKEN.ppo, gpu, gpuCount, mfu),
    },
  ];
  const pipelineTotal = stages
    .filter((s) => s.id !== "sftLora")
    .reduce((sum, s) => sum + s.usd, 0);

  return (
    <div id="calculator" className="grid gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
      {/* ---------------- controls ---------------- */}
      <div className="card p-5 sm:p-6 h-fit lg:sticky lg:top-6">
        <div className="eyebrow mb-4">configure</div>

        <div className="mb-5">
          <div className="text-[12.5px] mb-2" style={{ color: "var(--ink-secondary)" }}>
            model size
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  setPresetId(p.id);
                  setMfuOverride(null);
                }}
                className="rounded-md py-2 text-[12px] mono transition-colors"
                style={{
                  background: p.id === presetId ? "var(--accent)" : "var(--surface-2)",
                  color: p.id === presetId ? "#fff" : "var(--ink-secondary)",
                  border: `1px solid ${p.id === presetId ? "var(--accent)" : "var(--hairline)"}`,
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <p className="text-[11px] mt-2" style={{ color: "var(--ink-muted)" }}>
            {preset.layers} layers · {preset.hidden} hidden · {preset.heads} heads
            {preset.note ? ` · ${preset.note}` : ""}
          </p>
        </div>

        <div className="grid gap-5">
          <Control
            label="tokens per parameter"
            value={`${dataRatio}× → ${fmtTokens(uniqueTokens)}`}
            hint={
              dataRatio === CHINCHILLA_RATIO
                ? "20× is Chinchilla compute-optimal"
                : dataRatio > CHINCHILLA_RATIO
                  ? "over-trained: worse compute efficiency, better model per parameter"
                  : "under-trained: leaves capability on the table"
            }
          >
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={dataRatio}
              onChange={(e) => setDataRatio(+e.target.value)}
            />
          </Control>

          <Control
            label="epochs"
            value={String(epochs)}
            hint={`${fmtTokens(result.eff)} effective tokens after repetition decay`}
          >
            <input
              type="range"
              min={1}
              max={12}
              step={1}
              value={epochs}
              onChange={(e) => setEpochs(+e.target.value)}
            />
          </Control>

          <div>
            <div className="text-[12.5px] mb-2" style={{ color: "var(--ink-secondary)" }}>
              accelerator
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {GPUS.map((g) => (
                <button
                  key={g.id}
                  onClick={() => setGpuId(g.id)}
                  className="rounded-md py-2 text-[12px] mono transition-colors"
                  style={{
                    background: g.id === gpuId ? "var(--accent)" : "var(--surface-2)",
                    color: g.id === gpuId ? "#fff" : "var(--ink-secondary)",
                    border: `1px solid ${g.id === gpuId ? "var(--accent)" : "var(--hairline)"}`,
                  }}
                >
                  {g.name}
                </button>
              ))}
            </div>
            <p className="text-[11px] mt-2" style={{ color: "var(--ink-muted)" }}>
              ${gpu.usdPerHour.toFixed(2)}/GPU-hr · {(gpu.peakFlops / 1e12).toFixed(0)} TFLOP/s bf16
            </p>
          </div>

          <Control label="GPU count" value={String(gpuCount)}>
            <input
              type="range"
              min={1}
              max={512}
              step={1}
              value={gpuCount}
              onChange={(e) => setGpuCount(+e.target.value)}
            />
          </Control>

          <Control
            label="MFU"
            value={`${(mfu * 100).toFixed(0)}%`}
            hint={
              mfuOverride === null
                ? `default for ${fmtParams(N)} — our 125M run measured 22.5% end-to-end`
                : "manual override"
            }
          >
            <input
              type="range"
              min={10}
              max={60}
              step={1}
              value={Math.round(mfu * 100)}
              onChange={(e) => setMfuOverride(+e.target.value / 100)}
            />
          </Control>
        </div>

        {memWarn && (
          <div
            className="mt-5 rounded-lg px-3 py-2.5 text-[11.5px] leading-relaxed flex gap-2"
            style={{ background: "rgba(250,178,25,0.10)", border: "1px solid rgba(250,178,25,0.3)" }}
          >
            <span aria-hidden style={{ color: "var(--warning)" }}>
              ▲
            </span>
            <span style={{ color: "var(--ink-secondary)" }}>
              <span style={{ color: "var(--warning)" }} className="font-medium">
                Warning:
              </span>{" "}
              optimiser state alone needs ≈{memFloor} {gpu.name} at 16 bytes/param. At{" "}
              {gpuCount} you would need offload or heavier sharding, and the real MFU
              would fall below this estimate.
            </span>
          </div>
        )}
      </div>

      {/* ---------------- results ---------------- */}
      <div className="grid gap-6">
        <div className="card p-5 sm:p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3 mb-6">
            <div>
              <div className="eyebrow mb-2">predicted validation perplexity</div>
              <motion.div
                key={result.ppl.toFixed(3)}
                initial={{ opacity: 0.55 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="text-[46px] sm:text-[58px] font-semibold leading-none tracking-tight tnum"
                style={{ color: "var(--series-1)" }}
              >
                {result.ppl.toFixed(2)}
              </motion.div>
              <div className="mono text-[12px] mt-2" style={{ color: "var(--ink-muted)" }}>
                loss {result.loss.toFixed(4)} nats · vs {RUN.valPpl.toFixed(2)} measured at 125M
              </div>
            </div>
            <div className="text-right">
              <div className="eyebrow mb-2">compute cost</div>
              <motion.div
                key={result.usd.toFixed(2)}
                initial={{ opacity: 0.55 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="text-[46px] sm:text-[58px] font-semibold leading-none tracking-tight tnum"
              >
                {fmtUsd(result.usd)}
              </motion.div>
              <div className="mono text-[12px] mt-2" style={{ color: "var(--ink-muted)" }}>
                {fmtHours(result.wallHours)} wall on {gpuCount}× {gpu.name}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-px rounded-lg overflow-hidden" style={{ background: "var(--hairline)" }}>
            {[
              { k: "total FLOPs", v: fmtFlops(result.flops) },
              { k: "GPU-hours", v: Math.round(result.gpuHours).toLocaleString() },
              { k: "tokens processed", v: fmtTokens(result.nominal) },
              { k: "effective tokens", v: fmtTokens(result.eff) },
            ].map((s) => (
              <div key={s.k} className="px-3.5 py-3.5" style={{ background: "var(--surface-2)" }}>
                <div className="eyebrow mb-1.5">{s.k}</div>
                <div className="mono text-[14px] font-semibold">{s.v}</div>
              </div>
            ))}
          </div>

          <div className="grid sm:grid-cols-2 gap-6 mt-7">
            <MiniChart
              points={sweep.map((s) => ({ x: s.x, y: s.ppl }))}
              markIndex={epochs - 1}
              label="perplexity vs epochs"
              format={(n) => n.toFixed(1)}
              color="var(--series-1)"
              invertBetter
            />
            <MiniChart
              points={sweep.map((s) => ({ x: s.x, y: s.usd }))}
              markIndex={epochs - 1}
              label="cost vs epochs"
              format={(n) => fmtUsd(n)}
              color="var(--series-3)"
            />
          </div>

          <div
            className="mt-6 rounded-lg px-4 py-3.5 text-[12.5px] leading-relaxed"
            style={{ background: "var(--accent-soft)", border: "1px solid rgba(57,135,229,0.25)" }}
          >
            <span className="font-medium">Marginal epoch {epochs + 1}:</span> buys{" "}
            <span className="mono">{deltaPpl > 0.001 ? `−${deltaPpl.toFixed(3)}` : "≈0"}</span>{" "}
            perplexity for{" "}
            <span className="mono">{fmtUsd(deltaUsd)}</span>
            {deltaPpl > 0.001 ? (
              <>
                {" "}
                — <span className="mono">{fmtUsd(deltaUsd / deltaPpl)}</span> per point of
                perplexity.
              </>
            ) : (
              <> — effectively nothing. You are paying full compute for tokens the model has memorised.</>
            )}
          </div>
        </div>

        {/* full pipeline */}
        <div className="card p-5 sm:p-6">
          <h3 className="text-[15px] font-medium">Full pipeline at {fmtParams(N)}</h3>
          <p className="text-[12.5px] mt-1 mb-5" style={{ color: "var(--ink-muted)" }}>
            Every stage on {gpuCount}× {gpu.name} at {(mfu * 100).toFixed(0)}% MFU
          </p>

          <div className="scroll-x">
            <table className="w-full text-[12.5px] border-collapse min-w-[520px]">
              <thead>
                <tr style={{ color: "var(--ink-muted)" }} className="eyebrow">
                  <th className="text-left font-normal pb-2.5">stage</th>
                  <th className="text-right font-normal pb-2.5">FLOPs</th>
                  <th className="text-right font-normal pb-2.5">wall</th>
                  <th className="text-right font-normal pb-2.5">cost</th>
                </tr>
              </thead>
              <tbody>
                {stages.map((s) => (
                  <tr key={s.id} style={{ borderTop: "1px solid var(--hairline)" }}>
                    <td className="py-3 pr-4">
                      <div className={s.id === "sftLora" ? "" : "font-medium"}>
                        {s.name}
                        {s.id === "sftLora" ? (
                          <span
                            className="ml-2 text-[10px] mono px-1.5 py-0.5 rounded"
                            style={{ background: "var(--surface-2)", color: "var(--ink-muted)" }}
                          >
                            alternative
                          </span>
                        ) : null}
                      </div>
                      <div className="mono text-[11px] mt-0.5" style={{ color: "var(--ink-muted)" }}>
                        {s.detail}
                      </div>
                    </td>
                    <td className="py-3 text-right mono" style={{ color: "var(--ink-secondary)" }}>
                      {fmtFlops(s.flops)}
                    </td>
                    <td className="py-3 text-right mono" style={{ color: "var(--ink-secondary)" }}>
                      {fmtHours(s.wallHours)}
                    </td>
                    <td className="py-3 text-right mono font-semibold">{fmtUsd(s.usd)}</td>
                  </tr>
                ))}
                <tr style={{ borderTop: "1px solid var(--hairline-strong)" }}>
                  <td className="py-3 font-medium">
                    Total
                    <span className="mono text-[11px] ml-2" style={{ color: "var(--ink-muted)" }}>
                      pretrain + full SFT + DPO + PPO
                    </span>
                  </td>
                  <td />
                  <td />
                  <td className="py-3 text-right mono font-semibold" style={{ color: "var(--series-1)" }}>
                    {fmtUsd(pipelineTotal)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-muted)" }}>
            Post-training is rounding error against pretraining — which is why almost
            everyone fine-tunes someone else&apos;s base model. The SFT and preference
            datasets here are sized as fractions of the pretraining corpus (0.5% and 0.1%);
            in practice they are set by how much human annotation you can afford, and that
            labelling cost usually exceeds the GPU cost by an order of magnitude. None of
            the figures below include data acquisition, annotation, failed runs, or
            engineering time.
          </p>
        </div>
      </div>
    </div>
  );
}
