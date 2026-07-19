"use client";

import { useEffect, useState } from "react";

const LINKS = [
  { id: "playground", label: "Playground" },
  { id: "architecture", label: "Architecture" },
  { id: "training", label: "The run" },
  { id: "calculator", label: "Calculator" },
  { id: "method", label: "Method" },
];

export function Nav() {
  const [active, setActive] = useState("playground");
  const [solid, setSolid] = useState(false);

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 80);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    // rootMargin biases the "current" section toward the upper third of the
    // viewport, so the highlight changes when a heading reaches reading
    // position rather than when it barely clips the bottom edge.
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(e.target.id);
        }
      },
      { rootMargin: "-25% 0px -65% 0px" },
    );
    for (const l of LINKS) {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    }

    return () => {
      window.removeEventListener("scroll", onScroll);
      obs.disconnect();
    };
  }, []);

  return (
    <nav
      className="sticky top-0 z-50 transition-colors"
      style={{
        background: solid ? "rgba(11,13,18,0.82)" : "transparent",
        backdropFilter: solid ? "blur(12px)" : "none",
        WebkitBackdropFilter: solid ? "blur(12px)" : "none",
        borderBottom: `1px solid ${solid ? "var(--hairline)" : "transparent"}`,
      }}
    >
      <div className="mx-auto w-full max-w-[1120px] px-5 h-14 flex items-center justify-between gap-4">
        <a href="#top" className="mono text-[13px] font-semibold shrink-0">
          SLM-125M
        </a>

        <div className="scroll-x flex items-center gap-1 min-w-0">
          {LINKS.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              className="px-2.5 py-1.5 rounded-md text-[12.5px] whitespace-nowrap transition-colors"
              style={{
                color: active === l.id ? "var(--ink-primary)" : "var(--ink-muted)",
                background: active === l.id ? "var(--surface-2)" : "transparent",
              }}
            >
              {l.label}
            </a>
          ))}
        </div>

        <a
          href="https://huggingface.co/abhishekai/slm-125m-legal-base"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[12.5px] shrink-0 hidden sm:block"
          style={{ color: "var(--series-1)" }}
        >
          HuggingFace →
        </a>
      </div>
    </nav>
  );
}
