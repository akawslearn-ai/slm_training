"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useRef, useState } from "react";

/* The same prompts sample_app.py uses, so the demo matches the repo's own
   smoke test. All five are completion-shaped, never instruction-shaped -- this
   is a base model and an imperative prompt makes it look broken. */
const EXAMPLES = [
  "The plaintiff shall bear the burden of",
  "The Company's net revenues for the fiscal year",
  "IT IS HEREBY ORDERED that the motion",
  "Item 7. Management's Discussion and Analysis",
  "The court held that the defendant",
  "Pursuant to Section 13(a) of the Securities Exchange Act",
];

type Status = "idle" | "waking" | "streaming" | "error";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 mb-1">
        <label className="text-[12px]" style={{ color: "var(--ink-secondary)" }}>
          {label}
        </label>
        <span className="mono text-[12px] font-semibold">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        aria-label={label}
      />
      <p className="text-[10.5px] mt-0.5" style={{ color: "var(--ink-muted)" }}>
        {hint}
      </p>
    </div>
  );
}

export function Playground() {
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [tokenCount, setTokenCount] = useState(0);

  const [maxTokens, setMaxTokens] = useState(90);
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.95);
  const [topK, setTopK] = useState(50);

  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
  }, []);

  const run = useCallback(async () => {
    if (!prompt.trim()) return;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setOutput("");
    setError(null);
    setTokenCount(0);
    setElapsed(0);
    setStatus("waking");

    const t0 = performance.now();
    const timer = setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          top_p: topP,
          top_k: topK,
        }),
        signal: ac.signal,
      });

      if (!res.ok) {
        const j = await res.json().catch(() => ({ error: `Request failed (${res.status}).` }));
        throw new Error(j.error ?? `Request failed (${res.status}).`);
      }
      if (!res.body) throw new Error("No response stream.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let n = 0;

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          let payload: { token?: string; done?: boolean; error?: string };
          try {
            payload = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }
          if (payload.error) throw new Error(payload.error);
          if (payload.token) {
            n += 1;
            setTokenCount(n);
            setStatus("streaming");
            setOutput((o) => o + payload.token);
          }
        }
      }
      setStatus("idle");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError((e as Error).message);
      setStatus("error");
    } finally {
      clearInterval(timer);
      setElapsed((performance.now() - t0) / 1000);
      abortRef.current = null;
    }
  }, [prompt, maxTokens, temperature, topP, topK]);

  const busy = status === "waking" || status === "streaming";
  const tps = elapsed > 0 && tokenCount > 0 ? tokenCount / elapsed : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,300px)]">
      {/* ---- prompt + output ---- */}
      <div className="card p-5 sm:p-6">
        <div className="eyebrow mb-3">prompt</div>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
          }}
          rows={3}
          spellCheck={false}
          className="w-full resize-y rounded-lg px-3.5 py-3 text-[14px] leading-relaxed outline-none"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--hairline)",
            color: "var(--ink-primary)",
            fontFamily: "var(--font-mono)",
          }}
          placeholder="Write the opening of a sentence and the model will continue it…"
        />

        <div className="flex flex-wrap items-center gap-2 mt-3">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setPrompt(ex)}
              disabled={busy}
              className="text-[11.5px] px-2.5 py-1.5 rounded-md transition-colors text-left disabled:opacity-40"
              style={{
                background: prompt === ex ? "var(--accent-soft)" : "var(--surface-2)",
                border: `1px solid ${prompt === ex ? "rgba(57,135,229,0.4)" : "var(--hairline)"}`,
                color: prompt === ex ? "var(--ink-primary)" : "var(--ink-secondary)",
              }}
            >
              {ex.length > 42 ? ex.slice(0, 42) + "…" : ex}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3 mt-4">
          <button
            onClick={busy ? stop : run}
            disabled={!prompt.trim()}
            className="inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-[13.5px] font-medium transition-transform hover:-translate-y-px disabled:opacity-40 disabled:hover:translate-y-0"
            style={{
              background: busy ? "var(--surface-2)" : "var(--accent)",
              border: `1px solid ${busy ? "var(--hairline-strong)" : "var(--accent)"}`,
              color: busy ? "var(--ink-secondary)" : "#fff",
            }}
          >
            {busy ? "Stop" : "Generate"}
            {!busy && (
              <span className="mono text-[10.5px]" style={{ opacity: 0.7 }}>
                ⌘↵
              </span>
            )}
          </button>

          <AnimatePresence mode="wait">
            {status === "waking" && (
              <motion.span
                key="waking"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[12px] flex items-center gap-2"
                style={{ color: "var(--ink-muted)" }}
              >
                <span
                  className="inline-block h-[6px] w-[6px] rounded-full"
                  style={{ background: "var(--warning)", animation: "pulse-soft 1.2s ease-in-out infinite" }}
                />
                {elapsed > 3
                  ? `Waking the model — cold start takes ~20s (${elapsed.toFixed(0)}s)`
                  : "Connecting…"}
              </motion.span>
            )}
            {status === "streaming" && (
              <motion.span
                key="streaming"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mono text-[11.5px]"
                style={{ color: "var(--ink-muted)" }}
              >
                {tokenCount} tokens · {tps.toFixed(1)}/s
              </motion.span>
            )}
            {status === "idle" && tokenCount > 0 && (
              <motion.span
                key="done"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mono text-[11.5px]"
                style={{ color: "var(--ink-muted)" }}
              >
                {tokenCount} tokens in {elapsed.toFixed(1)}s · {tps.toFixed(1)}/s
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        {/* ---- output ---- */}
        <div className="eyebrow mt-6 mb-2">completion</div>
        <div
          className="rounded-lg px-4 py-4 min-h-[168px] text-[14px] leading-[1.75]"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--hairline)",
            fontFamily: "var(--font-mono)",
          }}
        >
          {error ? (
            <div className="flex gap-2 text-[13px]" style={{ color: "var(--critical)" }}>
              <span aria-hidden>▲</span>
              <span>{error}</span>
            </div>
          ) : output || busy ? (
            <>
              {/* The prompt stays visually distinct from what the model wrote,
                  so nobody mistakes their own words for a generation. */}
              <span style={{ color: "var(--ink-muted)" }}>{prompt}</span>
              <span>{output}</span>
              {busy && (
                <span
                  className="inline-block w-[7px] h-[15px] ml-[2px] align-middle"
                  style={{ background: "var(--accent)", animation: "pulse-soft 1s ease-in-out infinite" }}
                />
              )}
            </>
          ) : (
            <span style={{ color: "var(--ink-muted)" }}>
              Pick an example or write your own opening, then hit Generate.
            </span>
          )}
        </div>

        <p className="mt-3 text-[11.5px] leading-relaxed" style={{ color: "var(--ink-muted)" }}>
          Runs on a scale-to-zero CPU container, so the first request after a quiet spell
          pays a cold start. Output is a statistical continuation — names, citations,
          figures and holdings are fabricated.
        </p>
      </div>

      {/* ---- sampling controls ---- */}
      <div className="card p-5 sm:p-6 h-fit">
        <div className="eyebrow mb-4">sampling</div>
        <div className="grid gap-5">
          <Slider
            label="temperature"
            value={temperature}
            min={0}
            max={1.5}
            step={0.05}
            onChange={setTemperature}
            hint={temperature === 0 ? "greedy — deterministic" : "higher = more surprising"}
          />
          <Slider
            label="max new tokens"
            value={maxTokens}
            min={10}
            max={200}
            step={10}
            onChange={setMaxTokens}
            hint={`~${(maxTokens / 6.5).toFixed(0)}s at current CPU throughput`}
          />
          <Slider
            label="top-p"
            value={topP}
            min={0.1}
            max={1}
            step={0.05}
            onChange={setTopP}
            hint="nucleus: keep the smallest set summing to p"
          />
          <Slider
            label="top-k"
            value={topK}
            min={0}
            max={200}
            step={5}
            onChange={setTopK}
            hint={topK === 0 ? "disabled" : `sample from the ${topK} likeliest tokens`}
          />
        </div>

        <div
          className="mt-5 rounded-lg px-3 py-2.5 text-[11.5px] leading-relaxed"
          style={{ background: "var(--surface-2)", border: "1px solid var(--hairline)" }}
        >
          <span className="font-medium" style={{ color: "var(--ink-secondary)" }}>
            This is a base completer.
          </span>{" "}
          <span style={{ color: "var(--ink-muted)" }}>
            It continues text; it does not answer questions or follow instructions. Give it
            the start of a sentence, not a request.
          </span>
        </div>
      </div>
    </div>
  );
}
