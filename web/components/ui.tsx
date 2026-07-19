"use client";

import { motion, useInView, useMotionValue, useSpring, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

/** Fade + rise on scroll into view. Respects prefers-reduced-motion. */
export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

/**
 * Number that springs to its value when scrolled into view, and re-animates
 * whenever the value changes afterwards (so calculator outputs feel live).
 */
export function Counter({
  value,
  format,
  className = "",
}: {
  value: number;
  format: (n: number) => string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const reduce = useReducedMotion();
  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 90, damping: 20, mass: 0.6 });
  const [display, setDisplay] = useState(() => format(0));

  useEffect(() => {
    if (reduce) {
      setDisplay(format(value));
      return;
    }
    if (inView) mv.set(value);
  }, [inView, value, mv, reduce, format]);

  useEffect(() => {
    if (reduce) return;
    return spring.on("change", (v) => setDisplay(format(v)));
  }, [spring, format, reduce]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}

export function StatTile({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="card px-4 py-4 sm:px-5 sm:py-5">
      <div className="eyebrow mb-2">{label}</div>
      <div
        className="text-[26px] sm:text-[32px] font-semibold leading-none tracking-tight"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      {sub ? (
        <div className="mt-2 text-[12.5px] leading-snug" style={{ color: "var(--ink-muted)" }}>
          {sub}
        </div>
      ) : null}
    </div>
  );
}

export function Section({
  id,
  eyebrow,
  title,
  lede,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  lede?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="relative z-[1] mx-auto w-full max-w-[1120px] px-5 py-16 sm:py-24">
      <Reveal>
        <div className="eyebrow mb-3">{eyebrow}</div>
        <h2 className="text-[28px] sm:text-[38px] font-semibold tracking-tight leading-[1.12] max-w-[22ch]">
          {title}
        </h2>
        {lede ? (
          <p
            className="mt-4 max-w-[68ch] text-[15px] sm:text-[16.5px] leading-relaxed"
            style={{ color: "var(--ink-secondary)" }}
          >
            {lede}
          </p>
        ) : null}
      </Reveal>
      <div className="mt-9">{children}</div>
    </section>
  );
}

/** Inline note for caveats that must travel with the number they qualify. */
export function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="mt-4 text-[12.5px] leading-relaxed max-w-[76ch]"
      style={{ color: "var(--ink-muted)" }}
    >
      {children}
    </p>
  );
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-block h-[9px] w-[9px] rounded-full shrink-0"
            style={{ background: it.color }}
          />
          <span className="text-[12.5px]" style={{ color: "var(--ink-secondary)" }}>
            {it.label}
          </span>
        </div>
      ))}
    </div>
  );
}
