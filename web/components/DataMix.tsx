"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { DATA_MIX, RUN } from "@/lib/run-data";
import { Legend } from "./ui";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function DataMix() {
  const reduce = useReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const total = DATA_MIX.reduce((s, d) => s + d.tokens, 0);

  return (
    <div className="card p-5 sm:p-6">
      <h3 className="text-[15px] font-medium">Corpus composition</h3>
      <p className="text-[12.5px] mt-1 mb-5" style={{ color: "var(--ink-muted)" }}>
        {(RUN.trainTokens / 1e9).toFixed(2)}B training tokens, legal-first
      </p>

      {/* Stacked bar. 2px surface gap between segments per mark spec. */}
      <div className="flex gap-[2px] h-11 rounded-md overflow-hidden mb-5">
        {DATA_MIX.map((d, i) => (
          <motion.div
            key={d.name}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            className="relative first:rounded-l-md last:rounded-r-md"
            style={{ background: COLORS[i], opacity: hover === null || hover === i ? 1 : 0.45 }}
            initial={reduce ? { flexGrow: d.share } : { flexGrow: 0 }}
            whileInView={{ flexGrow: d.share }}
            viewport={{ once: true }}
            transition={{ duration: 0.9, delay: i * 0.12, ease: [0.22, 1, 0.36, 1] }}
          />
        ))}
      </div>

      <Legend items={DATA_MIX.map((d, i) => ({ color: COLORS[i], label: d.label }))} />

      <div className="mt-5 grid gap-0">
        {DATA_MIX.map((d, i) => (
          <div
            key={d.name}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            className="flex items-baseline justify-between gap-3 py-2.5 transition-colors"
            style={{
              borderTop: "1px solid var(--hairline)",
              background: hover === i ? "rgba(255,255,255,0.03)" : "transparent",
            }}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span
                aria-hidden
                className="h-[9px] w-[9px] rounded-full shrink-0"
                style={{ background: COLORS[i] }}
              />
              <span className="text-[13px] truncate">{d.label}</span>
              <span
                className="text-[10.5px] mono px-1.5 py-0.5 rounded shrink-0"
                style={{ background: "var(--surface-2)", color: "var(--ink-muted)" }}
              >
                {d.license}
              </span>
            </div>
            <div className="text-right shrink-0">
              <span className="mono text-[13px] font-semibold">
                {(d.share * 100).toFixed(1)}%
              </span>
              <span className="mono text-[11px] block" style={{ color: "var(--ink-muted)" }}>
                {(d.tokens / 1e6).toFixed(0)}M tokens
              </span>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-5 text-[12px] leading-relaxed" style={{ color: "var(--ink-muted)" }}>
        Not the usual web-heavy 70/20/10. The two legal sources cap out around 2B tokens,
        so the pipeline takes all of both and caps web at 0.5B — the mix is a consequence
        of what exists, not a target. Deduplicated with MinHash and decontaminated
        against LexGLUE/CaseHOLD with 13-gram matching before packing.
      </p>
    </div>
  );
}
