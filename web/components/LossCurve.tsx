"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useMemo, useRef, useState } from "react";
import { LOSS_CURVE, RUN } from "@/lib/run-data";

const W = 760;
const H = 320;
const PAD = { top: 18, right: 20, bottom: 38, left: 46 };

export function LossCurve() {
  const reduce = useReducedMotion();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const { path, pts, xScale, yScale, yTicks, xTicks } = useMemo(() => {
    const maxTok = RUN.tokensSeen;
    const losses = LOSS_CURVE.map((p) => p.loss);
    const yMin = Math.floor(Math.min(...losses) * 2) / 2 - 0.25;
    const yMax = Math.ceil(Math.max(...losses));

    const xScale = (t: number) =>
      PAD.left + (t / maxTok) * (W - PAD.left - PAD.right);
    const yScale = (l: number) =>
      PAD.top + (1 - (l - yMin) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);

    const pts = LOSS_CURVE.map((p) => ({ ...p, x: xScale(p.tokens), y: yScale(p.loss) }));
    const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");

    const yTicks: number[] = [];
    for (let v = Math.ceil(yMin); v <= yMax; v += 1) yTicks.push(v);
    const xTicks = [0, 1e9, 2e9, 3e9, 4e9].filter((t) => t <= maxTok);

    return { path, pts, xScale, yScale, yTicks, xTicks };
  }, []);

  const epochBoundary = xScale(RUN.trainTokens);
  const active = hoverIdx !== null ? pts[hoverIdx] : null;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const d = Math.abs(pts[i].x - px);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    setHoverIdx(best);
  }

  return (
    <div className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-1">
        <div>
          <h3 className="text-[15px] font-medium">Training loss</h3>
          <p className="text-[12.5px] mt-1" style={{ color: "var(--ink-muted)" }}>
            Measured, 7,778 steps over 4.08B tokens
          </p>
        </div>
        <button
          onClick={() => setShowTable((s) => !s)}
          className="text-[11.5px] mono px-2.5 py-1.5 rounded-md transition-colors"
          style={{ border: "1px solid var(--hairline-strong)", color: "var(--ink-secondary)" }}
        >
          {showTable ? "chart" : "table"}
        </button>
      </div>

      {showTable ? (
        <div className="scroll-x mt-4 max-h-[320px] overflow-y-auto">
          <table className="w-full text-[12px] mono border-collapse">
            <thead className="sticky top-0" style={{ background: "var(--surface-1)" }}>
              <tr style={{ color: "var(--ink-muted)" }}>
                <th className="text-left font-normal py-2 pr-4">step</th>
                <th className="text-right font-normal py-2 pr-4">tokens</th>
                <th className="text-right font-normal py-2">loss</th>
              </tr>
            </thead>
            <tbody>
              {LOSS_CURVE.map((p) => (
                <tr key={p.step} style={{ borderTop: "1px solid var(--hairline)" }}>
                  <td className="py-1.5 pr-4">{p.step.toLocaleString()}</td>
                  <td className="py-1.5 pr-4 text-right" style={{ color: "var(--ink-secondary)" }}>
                    {(p.tokens / 1e9).toFixed(3)}B
                  </td>
                  <td className="py-1.5 text-right">{p.loss.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="relative mt-3">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="w-full h-auto block"
            onMouseMove={onMove}
            onMouseLeave={() => setHoverIdx(null)}
            role="img"
            aria-label={`Training loss falling from 8.56 to ${LOSS_CURVE[LOSS_CURVE.length - 1].loss.toFixed(2)} nats over 4.08 billion tokens`}
          >
            {/* gridlines */}
            {yTicks.map((v) => (
              <g key={v}>
                <line
                  x1={PAD.left}
                  x2={W - PAD.right}
                  y1={yScale(v)}
                  y2={yScale(v)}
                  stroke="var(--grid)"
                  strokeWidth="1"
                />
                <text
                  x={PAD.left - 10}
                  y={yScale(v)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize="10.5"
                  fill="var(--ink-muted)"
                  className="mono"
                >
                  {v.toFixed(1)}
                </text>
              </g>
            ))}

            {xTicks.map((t) => (
              <text
                key={t}
                x={xScale(t)}
                y={H - PAD.bottom + 18}
                textAnchor="middle"
                fontSize="10.5"
                fill="var(--ink-muted)"
                className="mono"
              >
                {(t / 1e9).toFixed(0)}B
              </text>
            ))}

            {/* epoch boundary */}
            <line
              x1={epochBoundary}
              x2={epochBoundary}
              y1={PAD.top}
              y2={H - PAD.bottom}
              stroke="var(--hairline-strong)"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
            <text
              x={epochBoundary + 6}
              y={PAD.top + 11}
              fontSize="10"
              fill="var(--ink-muted)"
              className="mono"
            >
              epoch 2
            </text>

            {/* baseline */}
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={H - PAD.bottom}
              y2={H - PAD.bottom}
              stroke="var(--axis)"
              strokeWidth="1"
            />

            {/* the curve */}
            <motion.path
              d={path}
              fill="none"
              stroke="var(--series-1)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={reduce ? undefined : { pathLength: 0 }}
              whileInView={reduce ? undefined : { pathLength: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.6, ease: "easeOut" }}
            />

            {/* crosshair */}
            {active && (
              <g>
                <line
                  x1={active.x}
                  x2={active.x}
                  y1={PAD.top}
                  y2={H - PAD.bottom}
                  stroke="var(--hairline-strong)"
                  strokeWidth="1"
                />
                <circle
                  cx={active.x}
                  cy={active.y}
                  r="5"
                  fill="var(--series-1)"
                  stroke="var(--surface-1)"
                  strokeWidth="2"
                />
              </g>
            )}
          </svg>

          {/* tooltip */}
          {active && (
            <div
              className="pointer-events-none absolute card-2 px-3 py-2 text-[11.5px] mono whitespace-nowrap"
              style={{
                left: `${(active.x / W) * 100}%`,
                top: `${(active.y / H) * 100}%`,
                transform: `translate(${active.x > W * 0.62 ? "calc(-100% - 14px)" : "14px"}, -50%)`,
                boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
              }}
            >
              <div style={{ color: "var(--ink-muted)" }}>step {active.step.toLocaleString()}</div>
              <div className="mt-1">
                loss <span className="font-semibold">{active.loss.toFixed(4)}</span>
              </div>
              <div style={{ color: "var(--ink-secondary)" }}>
                {(active.tokens / 1e9).toFixed(2)}B tokens
              </div>
            </div>
          )}
        </div>
      )}

      <p className="mt-4 text-[12px] leading-relaxed" style={{ color: "var(--ink-muted)" }}>
        Loss falls from 8.56 (random init, ≈ uniform over 16K vocab) to 2.20. The
        visible step down at the epoch boundary is the model seeing the corpus a second
        time. Final held-out perplexity, measured over the full 20.6M-token validation
        split, is <span style={{ color: "var(--ink-primary)" }}>{RUN.valPpl.toFixed(3)}</span>.
      </p>
    </div>
  );
}
