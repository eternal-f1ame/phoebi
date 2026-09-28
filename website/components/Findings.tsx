import { findings, sections } from "@/content";
import SectionHeading from "./SectionHeading";
import styles from "./Findings.module.css";

// The three findings that Figure 1 supports, in plain words.
export default function Findings() {
  return (
    <section id="findings" className="section">
      <div className="container">
        <SectionHeading {...sections.findings} />
        <ol className={styles.list}>
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
