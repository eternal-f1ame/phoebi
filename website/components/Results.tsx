import { sections, teaser } from "@/content";
import SectionHeading from "./SectionHeading";
import styles from "./Results.module.css";

// Figure 1 with the paper's caption, on one screen; the findings follow in their own section.
export default function Results() {
  return (
    <section id="results" className="section">
      <div className="container">
        <SectionHeading {...sections.results} />
        <figure className={`figure ${styles.figure}`}>
          <img src={teaser.image} alt={teaser.alt} width={teaser.width} height={teaser.height} loading="lazy" />
          <figcaption>
            <span className="figure-label">Figure 1.</span> {teaser.caption}
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
