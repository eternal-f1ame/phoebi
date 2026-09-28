import { leads, sections, species } from "@/content";
import Eyepiece from "./Eyepiece";
import SectionHeading from "./SectionHeading";
import styles from "./Species.module.css";

// A specimen sheet: each species in its own eyepiece view with its reference data. The Gram
// result is marked by the colour of the stain it takes: violet (+) or safranin (−).
export default function Species() {
  return (
    <section id="species" className="section">
      <div className="container">
        <SectionHeading {...sections.species} />
        <p className="lead">{leads.species}</p>
        <div className={styles.grid}>
          {species.map((s) => (
            <article key={s.code} className={styles.card}>
              <div className={styles.view}>
                <Eyepiece src={s.image} alt={`Phase-contrast view of ${s.name}`} width={480} height={480} size="small" reticle={false} lazy />
              </div>
              <div>
                <p className={styles.code}>{s.code}</p>
                <h3 className={styles.name}>
                  <em>{s.name}</em>
                </h3>
                <dl className={styles.facts}>
                  <div>
                    <dt>Gram</dt>
                    <dd>
                      <span className={`${styles.dot} ${s.gram === "+" ? styles.positive : styles.negative}`} aria-hidden="true" />
                      {s.gram === "+" ? "positive" : "negative"}
                    </dd>
                  </div>
                  <div>
                    <dt>Motility</dt>
                    <dd>{s.motility}</dd>
                  </div>
                  <div>
                    <dt>Length</dt>
                    <dd>{s.length}</dd>
                  </div>
                </dl>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
