import type { CSSProperties } from "react";
import { findings, sections } from "@/content";
import SectionHeading from "./SectionHeading";
import styles from "./Findings.module.css";

// Column widths in proportion to each finding's length, so the cards fill evenly.
const widths = Object.fromEntries(findings.map((f, i) => [`--w${i + 1}`, `${f.title.length + f.text.length}fr`]));

// The three findings that Figure 1 supports, in plain words.
export default function Findings() {
  return (
    <section id="findings" className="section">
      <div className="container">
        <SectionHeading {...sections.findings} />
        <ol className={styles.list} style={widths as CSSProperties}>
          {findings.map((f, i) => (
            <li key={f.title}>
              <span className={styles.number} aria-hidden="true">
                {i + 1}
              </span>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
