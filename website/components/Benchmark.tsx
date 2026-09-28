import { leads, numbers, protocols, sections } from "@/content";
import Counter from "./Counter";
import LcoMatrix from "./LcoMatrix";
import SectionHeading from "./SectionHeading";
import styles from "./Benchmark.module.css";

export default function Benchmark() {
  return (
    <section id="benchmark" className="section">
      <div className="container">
        {/* the heading and the headline numbers share one band */}
        <div className={styles.top}>
          <SectionHeading {...sections.benchmark} />
          <ul className={styles.stats}>
            {numbers.map((n) => (
              <li key={n.label}>
                <span className={styles.value}>
                  <Counter value={n.value} />
                </span>
                <span className={styles.label}>{n.label}</span>
              </li>
            ))}
          </ul>
        </div>
        <p className="lead">{leads.benchmark}</p>
        <dl className={styles.protocols}>
          {protocols.map((p) => (
            <div key={p.name}>
              <dt>{p.name}</dt>
              <dd>{p.text}</dd>
            </div>
          ))}
        </dl>
        <LcoMatrix />
      </div>
    </section>
  );
}
