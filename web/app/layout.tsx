import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SLM-125M — a legal/financial language model, from scratch",
  description:
    "A 125M-parameter Llama-style model pretrained from random weights on US case law and SEC filings for $16.93. Architecture, measured training curve, and a calibrated scaling calculator.",
  openGraph: {
    title: "SLM-125M — a legal/financial language model, from scratch",
    description:
      "Pretrained from random weights for $16.93. Explore the architecture and extrapolate the cost of training bigger models.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
