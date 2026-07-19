"use client";

import { motion, useReducedMotion } from "framer-motion";
import { RUN } from "@/lib/run-data";
import { Counter } from "./ui";

const HF_URL = "https://huggingface.co/abhishekai/slm-125m-legal-base";

const STATS = [
  { label: "parameters", value: RUN.params / 1e6, fmt: (n: number) => `${n.toFixed(1)}M` },
  { label: "vocab", value: RUN.vocabSize, fmt: (n: number) => Math.round(n).toLocaleString() },
  { label: "context", value: RUN.contextLength, fmt: (n: number) => Math.round(n).toLocaleString() },
  { label: "train tokens", value: RUN.trainTokens / 1e9, fmt: (n: number) => `${n.toFixed(2)}B` },
  { label: "val perplexity", value: RUN.valPpl, fmt: (n: number) => n.toFixed(2) },
  { label: "total cost", value: RUN.pretrainUsd + RUN.dataUsd, fmt: (n: number) => `$${n.toFixed(2)}` },
];

export function Hero() {
  const reduce = useReducedMotion();

  return (
    <header className="relative z-[1] mx-auto w-full max-w-[1120px] px-5 pt-20 pb-10 sm:pt-32 sm:pb-16">
      <motion.div
        initial={{ opacity: 0, y: reduce ? 0 : 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="flex items-center gap-2.5 mb-6">
          <span
            className="inline-block h-[7px] w-[7px] rounded-full"
            style={{ background: "var(--good)", animation: reduce ? undefined : "pulse-soft 2.4s ease-in-out infinite" }}
          />
          <span className="eyebrow">base model · pretrained from random weights</span>
        </div>

        <h1 className="text-[40px] sm:text-[68px] font-semibold tracking-[-0.03em] leading-[1.02] max-w-[16ch]">
          SLM-125M
        </h1>

        <p
          className="mt-6 max-w-[62ch] text-[16px] sm:text-[19px] leading-relaxed"
          style={{ color: "var(--ink-secondary)" }}
        >
          A 125-million-parameter Llama-style language model, trained from scratch on US
          case law and SEC filings. No pretrained weights, no distillation — random
          initialisation to fluent legal prose for{" "}
          <span style={{ color: "var(--ink-primary)" }} className="font-medium">
            $16.93
          </span>{" "}
          of compute.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <a
            href={HF_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-[14px] font-medium transition-transform hover:-translate-y-px"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Model on HuggingFace
            <span aria-hidden>→</span>
          </a>
          <a
            href="#calculator"
            className="inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-[14px] font-medium transition-colors"
            style={{
              border: "1px solid var(--hairline-strong)",
              color: "var(--ink-secondary)",
            }}
          >
            Scaling calculator
          </a>
        </div>
      </motion.div>

      <motion.dl
        className="mt-14 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-px rounded-xl overflow-hidden"
        style={{ background: "var(--hairline)" }}
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: 0.06, delayChildren: 0.25 } } }}
      >
        {STATS.map((s) => (
          <motion.div
            key={s.label}
            className="px-4 py-5"
            style={{ background: "var(--surface-1)" }}
            variants={{
              hidden: { opacity: 0, y: reduce ? 0 : 10 },
              show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
            }}
          >
            <dt className="eyebrow mb-2">{s.label}</dt>
            <dd className="text-[21px] font-semibold tracking-tight tnum">
              <Counter value={s.value} format={s.fmt} />
            </dd>
          </motion.div>
        ))}
      </motion.dl>
    </header>
  );
}
