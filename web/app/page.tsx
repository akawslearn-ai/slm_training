import { Hero } from "@/components/Hero";
import { Nav } from "@/components/Nav";
import { Architecture } from "@/components/Architecture";
import { LossCurve } from "@/components/LossCurve";
import { DataMix } from "@/components/DataMix";
import { Calculator } from "@/components/Calculator";
import { Playground } from "@/components/Playground";
import { Section, Reveal, StatTile, Caveat } from "@/components/ui";
import { RUN } from "@/lib/run-data";
import { LOSS_MODEL, REPEAT_DECAY, CHINCHILLA_RATIO } from "@/lib/scaling";

const HF_URL = "https://huggingface.co/abhishekai/slm-125m-legal-base";

export default function Page() {
  return (
    <main id="top">
      <Nav />
      <Hero />

      <Section
        id="playground"
        eyebrow="01 — playground"
        title="Give it the start of a sentence"
        lede="Live generation from the model on this page. It is a base completer with no instruction tuning, so it continues text rather than answering it — the examples below are all sentence openings, which is the shape it was trained on."
      >
        <Playground />
      </Section>

      <Section
        id="architecture"
        eyebrow="02 — architecture"
        title="Twelve blocks, 768 dimensions, a 16K vocabulary"
        lede="A standard Llama decoder, shrunk until it fits a hobbyist budget. Every choice here is a consequence of the parameter count: the small vocabulary keeps the embedding table from eating the budget, tied weights save another 12.6M parameters, and multi-head attention is kept because grouped-query saves nothing worth having at this size."
      >
        <Architecture />
      </Section>

      <Section
        id="training"
        eyebrow="03 — the run"
        title="From random weights to fluent legal prose"
        lede="Two epochs over 2.04B tokens on 8×H100. The corpus was streamed, cleaned through a six-step deterministic chain, deduplicated with MinHash, and decontaminated against LexGLUE and CaseHOLD before a single gradient step."
      >
        <div className="grid gap-6">
          <LossCurve />
          <div className="grid gap-6 lg:grid-cols-2">
            <DataMix />
            <div className="grid gap-4 content-start">
              <div className="grid grid-cols-2 gap-4">
                <StatTile
                  label="val perplexity"
                  value={RUN.valPpl.toFixed(2)}
                  sub="full 20.6M-token validation split"
                  accent="var(--series-1)"
                />
                <StatTile
                  label="val loss"
                  value={RUN.valLoss.toFixed(3)}
                  sub="nats per token"
                />
                <StatTile
                  label="optimiser steps"
                  value={RUN.steps.toLocaleString()}
                  sub="524,288 tokens per step"
                />
                <StatTile
                  label="throughput"
                  value="411K"
                  sub="tokens/s per GPU, steady state"
                />
              </div>
              <div className="card p-5">
                <div className="eyebrow mb-3">what it cost</div>
                <div className="grid gap-2.5">
                  {[
                    { k: "Phases 0–4 · data pipeline", v: "$1.73", s: "CPU only" },
                    { k: "Phase 5 · pretraining", v: "$15.20", s: "8×H100, 2 epochs" },
                  ].map((r) => (
                    <div key={r.k} className="flex items-baseline justify-between gap-3">
                      <span className="text-[13px]" style={{ color: "var(--ink-secondary)" }}>
                        {r.k}
                        <span className="block text-[11px]" style={{ color: "var(--ink-muted)" }}>
                          {r.s}
                        </span>
                      </span>
                      <span className="mono text-[14px] font-semibold shrink-0">{r.v}</span>
                    </div>
                  ))}
                  <div
                    className="flex items-baseline justify-between gap-3 pt-2.5"
                    style={{ borderTop: "1px solid var(--hairline-strong)" }}
                  >
                    <span className="text-[13px] font-medium">Total</span>
                    <span
                      className="mono text-[19px] font-semibold"
                      style={{ color: "var(--series-1)" }}
                    >
                      $16.93
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <Caveat>
          Cost is GPU-seconds billed, including <span className="mono">torch.compile</span> and
          CUDA init — end-to-end MFU was 22.5% against 31–36% in steady state. It excludes
          the failed runs, the smoke tests, and every hour of engineering time, which is
          the honest majority of what building this actually took.
        </Caveat>
      </Section>

      <Section
        id="calculator"
        eyebrow="04 — extrapolation"
        title="What would a bigger one cost?"
        lede="The loss curve above is one measured point. Anchoring the Chinchilla scaling law to it lets you ask what the same recipe would produce at 1B, 13B or 50B parameters — how many epochs buy how much perplexity, on what hardware, for how much money."
      >
        <Calculator />
      </Section>

      <Section
        id="method"
        eyebrow="05 — methodology"
        title="How these numbers are produced"
        lede="Every figure in the calculator comes from three published relationships and one measured anchor. None of it is a black box, and all of it can be wrong in ways worth understanding."
      >
        <div className="grid gap-4 md:grid-cols-2">
          {[
            {
              h: "Loss",
              b: (
                <>
                  The Chinchilla parametric form{" "}
                  <span className="mono" style={{ color: "var(--ink-primary)" }}>
                    L = E + A/N^α + B/D^β
                  </span>{" "}
                  with Hoffmann et al.&apos;s published{" "}
                  <span className="mono">A={LOSS_MODEL.A}, α={LOSS_MODEL.alpha}, B={LOSS_MODEL.B},
                  β={LOSS_MODEL.beta}</span>. The irreducible term{" "}
                  <span className="mono">E</span> is re-solved to{" "}
                  <span className="mono">{LOSS_MODEL.E}</span> so the curve passes exactly
                  through our measured point, rather than Chinchilla&apos;s 1.69.
                </>
              ),
            },
            {
              h: "Compute",
              b: (
                <>
                  <span className="mono" style={{ color: "var(--ink-primary)" }}>
                    C = 6ND
                  </span>{" "}
                  for training — 2N forward, 4N backward. Wall time is{" "}
                  <span className="mono">C / (GPUs × peak × MFU)</span>. Post-training
                  stages change only the leading coefficient: LoRA 4N, DPO 16N (chosen and
                  rejected sequences through both policy and a frozen reference), PPO 36N.
                </>
              ),
            },
            {
              h: "Repeated data",
              b: (
                <>
                  Epochs past the first are discounted following Muennighoff et al. (2023),
                  with decay constant <span className="mono">R* ≈ {REPEAT_DECAY}</span>. Four
                  epochs are worth ~3.7 fresh ones; twelve are worth ~6.9. You pay full
                  compute for all of them, which is the whole point of the second chart.
                </>
              ),
            },
            {
              h: "Price",
              b: (
                <>
                  Modal on-demand: H100 80GB at $3.95/GPU-hr, A100 80GB at $2.50. Our own
                  run cost <span className="mono">$0.0049 per PFLOP</span> all-in, which is
                  the sanity check the rest is calibrated against. Reserved capacity and
                  spot pricing are both materially cheaper.
                </>
              ),
            },
          ].map((c, i) => (
            <Reveal key={c.h} delay={i * 0.06}>
              <div className="card p-5 h-full">
                <div className="text-[14px] font-medium mb-2.5">{c.h}</div>
                <p className="text-[12.5px] leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
                  {c.b}
                </p>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.1}>
          <div
            className="card p-5 mt-4"
            style={{ borderColor: "rgba(250,178,25,0.28)", background: "rgba(250,178,25,0.05)" }}
          >
            <div className="text-[14px] font-medium mb-2.5" style={{ color: "var(--warning)" }}>
              ▲ Where this stops being trustworthy
            </div>
            <ul
              className="text-[12.5px] leading-relaxed grid gap-2 list-disc pl-4"
              style={{ color: "var(--ink-secondary)" }}
            >
              <li>
                <span style={{ color: "var(--ink-primary)" }}>One anchor cannot fit two exponents.</span>{" "}
                We re-solved a single constant against a single run. The shape of the curve
                is borrowed from Chinchilla and assumes a legal/financial corpus scales like
                general web text. Treat 10× as indicative and 100× as a back-of-envelope.
              </li>
              <li>
                <span style={{ color: "var(--ink-primary)" }}>Perplexity is not capability.</span>{" "}
                It is tokenizer- and corpus-specific — a 16K vocabulary flatters these
                numbers against a 50K one, and none of it is comparable to a published
                perplexity on a different corpus. It says nothing about reasoning,
                instruction-following, or factual accuracy.
              </li>
              <li>
                <span style={{ color: "var(--ink-primary)" }}>MFU is assumed, not measured,</span>{" "}
                above 125M. Real large-scale runs lose throughput to communication,
                stragglers, checkpointing and restarts. At {CHINCHILLA_RATIO}× tokens per
                parameter and hundreds of GPUs, the gap between this estimate and reality
                widens considerably.
              </li>
              <li>
                <span style={{ color: "var(--ink-primary)" }}>Compute is not the budget.</span>{" "}
                These figures exclude data acquisition and licensing, human annotation,
                storage and egress, failed runs, and salaries. On real projects that
                remainder is usually the larger half.
              </li>
            </ul>
          </div>
        </Reveal>
      </Section>

      <Section
        id="limits"
        eyebrow="06 — honesty"
        title="What this model cannot do"
        lede="It is a 125M base model. It writes convincing legal and financial prose, which is precisely what makes its failures dangerous."
      >
        <div className="grid gap-4 md:grid-cols-3">
          {[
            {
              h: "No arithmetic",
              b: "It will state a revenue figure, a comparison figure, and a percentage change that do not follow from each other. Expected at this scale. Never treat a number it generates as a computation.",
            },
            {
              h: "Fabricates citations",
              b: "It reproduces the register of judicial opinions fluently, so invented case names, docket numbers and holdings look entirely plausible. It is not a source of legal fact and is not legal advice.",
            },
            {
              h: "Does not follow instructions",
              b: "This is a base completer, not a chat model. There is no SFT, no instruction tuning, no RLHF. Prompt it as a text continuation and it behaves; prompt it as an assistant and it will not.",
            },
          ].map((c, i) => (
            <Reveal key={c.h} delay={i * 0.06}>
              <div className="card p-5 h-full">
                <div className="text-[14px] font-medium mb-2.5">{c.h}</div>
                <p className="text-[12.5px] leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
                  {c.b}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      <footer
        className="relative z-[1] mx-auto w-full max-w-[1120px] px-5 py-12"
        style={{ borderTop: "1px solid var(--hairline)" }}
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="text-[12.5px]" style={{ color: "var(--ink-muted)" }}>
            SLM-125M · pretrained from random weights on Modal · {RUN.steps.toLocaleString()} steps
          </div>
          <a
            href={HF_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[12.5px] mono"
            style={{ color: "var(--series-1)" }}
          >
            huggingface.co/abhishekai/slm-125m-legal-base →
          </a>
        </div>
      </footer>
    </main>
  );
}
