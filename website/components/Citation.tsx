"use client";

import { useEffect, useRef, useState } from "react";
import { bibtex, sections } from "@/content";
import SectionHeading from "./SectionHeading";
import styles from "./Citation.module.css";

type Status = "idle" | "copied" | "manual";
const MESSAGE: Record<Status, string> = {
  idle: "",
  copied: "Copied to clipboard.",
  manual: "Selected. Press Ctrl+C (⌘C on a Mac) to copy.",
};

// Copies the BibTeX. Where the clipboard API is missing or refused (plain http, older
// browsers, denied permission) it selects the text instead and says how to copy it.
export default function Citation() {
  const pre = useRef<HTMLPreElement>(null);
  const [status, setStatus] = useState<Status>("idle");

  useEffect(() => {
    if (status !== "copied") return;
    const t = setTimeout(() => setStatus("idle"), 2000);
    return () => clearTimeout(t);
  }, [status]);

  async function copy() {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(bibtex);
      setStatus("copied");
    } catch {
      const el = pre.current;
      const selection = window.getSelection();
      if (el && selection) {
        const range = document.createRange();
        range.selectNodeContents(el);
        selection.removeAllRanges();
        selection.addRange(range);
      }
      setStatus("manual");
    }
  }

  return (
    <section id="citation" className="section">
      <div className="container">
        <SectionHeading {...sections.citation} />
        <div className={styles.box}>
          <button type="button" className={styles.copy} onClick={copy}>
            {status === "copied" ? "Copied" : "Copy"}
          </button>
          <pre ref={pre} className={styles.bib} tabIndex={0} aria-label="BibTeX entry">
            <code>{bibtex}</code>
          </pre>
        </div>
        <p className={styles.hint} aria-live="polite">
          {MESSAGE[status]}
        </p>
      </div>
    </section>
  );
}
