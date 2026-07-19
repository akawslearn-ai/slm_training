"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { RUN } from "@/lib/run-data";

const SPECS: { k: string; v: string; note?: string }[] = [
  { k: "architecture", v: "LlamaForCausalLM" },
  { k: "layers", v: String(RUN.layers) },
  { k: "hidden size", v: String(RUN.hidden) },
  { k: "attention heads", v: `${RUN.heads} (head dim 64)` },
  { k: "kv heads", v: `${RUN.kvHeads}`, note: "= heads, so multi-head, not grouped-query" },
  { k: "intermediate", v: `${RUN.intermediate}`, note: "SwiGLU inner dimension" },
  { k: "context length", v: RUN.contextLength.toLocaleString() },
  { k: "vocab", v: RUN.vocabSize.toLocaleString(), note: "byte-level BPE trained on this corpus" },
  { k: "position", v: "RoPE, theta 10,000" },
  { k: "norm", v: "RMSNorm, eps 1e-5" },
  { k: "embeddings", v: "tied", note: "input and output share one matrix" },
  { k: "attention bias", v: "none" },
];

/** One transformer block in the stack diagram. */
function Block({
  i,
  active,
  onHover,
}: {
  i: number;
  active: boolean;
  onHover: (i: number | null) => void;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      onMouseEnter={() => onHover(i)}
      onMouseLeave={() => onHover(null)}
      className="relative rounded-[5px] cursor-default"
      style={{
        height: 22,
        background: active ? "var(--accent)" : "var(--surface-2)",
        border: `1px solid ${active ? "var(--accent)" : "var(--hairline)"}`,
      }}
      initial={{ opacity: 0, x: reduce ? 0 : -12 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: reduce ? 0 : i * 0.035, ease: [0.22, 1, 0.36, 1] }}
    >
      <div
        className="absolute inset-y-0 left-3 flex items-center text-[10.5px] mono"
        style={{ color: active ? "#fff" : "var(--ink-muted)" }}
      >
        block {i}
      </div>
      <div
        className="absolute inset-y-0 right-3 flex items-center gap-2 text-[10px] mono"
        style={{ color: active ? "rgba(255,255,255,0.85)" : "var(--ink-muted)" }}
      >
        <span>attn</span>
        <span aria-hidden style={{ opacity: 0.4 }}>·</span>
        <span>mlp</span>
      </div>
    </motion.div>
  );
}

export function Architecture() {
  const [hover, setHover] = useState<number | null>(null);
  const reduce = useReducedMotion();

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      {/* Stack diagram */}
      <div className="card p-5 sm:p-6">
        <div className="eyebrow mb-4">forward pass</div>

        <div className="flex flex-col gap-[5px]">
          <div
            className="rounded-[5px] px-3 py-2 text-[11px] mono flex items-center justify-between"
            style={{ background: "var(--surface-2)", border: "1px solid var(--hairline)", color: "var(--ink-secondary)" }}
          >
            <span>token + rope embed</span>
            <span style={{ color: "var(--ink-muted)" }}>
              {RUN.vocabSize.toLocaleString()} × {RUN.hidden}
            </span>
          </div>

          {/* Flow connector */}
          <svg height="14" className="w-full" aria-hidden>
            <line
              x1="14"
              y1="0"
              x2="14"
              y2="14"
              stroke="var(--accent)"
              strokeWidth="1.5"
              className={reduce ? undefined : "flow-line"}
            />
          </svg>

          {Array.from({ length: RUN.layers }, (_, i) => (
            <Block key={i} i={i} active={hover === i} onHover={setHover} />
          ))}

          <svg height="14" className="w-full" aria-hidden>
            <line
              x1="14"
              y1="0"
              x2="14"
              y2="14"
              stroke="var(--accent)"
              strokeWidth="1.5"
              className={reduce ? undefined : "flow-line"}
            />
          </svg>

          <div
            className="rounded-[5px] px-3 py-2 text-[11px] mono flex items-center justify-between"
            style={{ background: "var(--surface-2)", border: "1px solid var(--hairline)", color: "var(--ink-secondary)" }}
          >
            <span>rmsnorm → lm head (tied)</span>
            <span style={{ color: "var(--ink-muted)" }}>
              {RUN.hidden} × {RUN.vocabSize.toLocaleString()}
            </span>
          </div>
        </div>

        <p className="mt-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-muted)" }}>
          Twelve identical blocks. Each runs multi-head attention over a 1,024-token
          window, then a SwiGLU feed-forward at 4× width. The embedding matrix is reused
          as the output head, which is why a 16K vocab matters at this scale — it would
          otherwise consume a fifth of the parameter budget.
        </p>
      </div>

      {/* Spec table */}
      <div className="card p-5 sm:p-6">
        <div className="eyebrow mb-4">configuration</div>
        <dl className="grid gap-0">
          {SPECS.map((s, i) => (
            <motion.div
              key={s.k}
              className="flex items-baseline justify-between gap-4 py-[9px]"
              style={{ borderTop: i === 0 ? "none" : "1px solid var(--hairline)" }}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: reduce ? 0 : i * 0.03 }}
            >
              <dt className="text-[13px] shrink-0" style={{ color: "var(--ink-secondary)" }}>
                {s.k}
              </dt>
              <dd className="text-right min-w-0">
                <span className="mono text-[12.5px]">{s.v}</span>
                {s.note ? (
                  <span className="block text-[11px] mt-0.5" style={{ color: "var(--ink-muted)" }}>
                    {s.note}
                  </span>
                ) : null}
              </dd>
            </motion.div>
          ))}
        </dl>
      </div>
    </div>
  );
}
