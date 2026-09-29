import { Fragment } from "react";
import { affiliation, authors, field, site } from "@/content";
import Eyepiece from "./Eyepiece";
import LinkButtons from "./LinkButtons";
import styles from "./Hero.module.css";

// The page header: the title block beside a real field from the six-species culture, seen as
// through an eyepiece. The title stays one heading, with its colon kept for screen readers.
export default function Hero() {
  return (
    <header className={styles.hero}>
      <div className={`container ${styles.grid}`}>
        <div>
          <p className={styles.venue}>
            <a href={site.venueUrl}>{site.venue}</a>
          </p>
          <h1 className={styles.title}>
            <span className={styles.name}>{site.short}</span>
            <span className="sr-only">:</span> <span className={styles.subtitle}>{site.subtitle}</span>
          </h1>
          <p className={styles.authors}>
            {authors.map((a, i) => (
              <Fragment key={a.email}>
                {i > 0 && " "}
                <span className={styles.author}>
                  <a href={`mailto:${a.email}`}>{a.name}</a>
                </span>
              </Fragment>
            ))}
          </p>
          <p className={styles.affiliation}>{affiliation}</p>
          <LinkButtons />
        </div>
        <figure className={styles.view}>
          <Eyepiece src={field.image} alt={field.alt} width={field.width} height={field.height} focus />
          <figcaption>{field.caption}</figcaption>
        </figure>
      </div>
    </header>
  );
}
