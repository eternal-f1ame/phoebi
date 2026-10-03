import type { ReactNode } from "react";
import { links } from "@/content";
import styles from "./LinkButtons.module.css";

const icon = (path: ReactNode) => (
  <svg className={styles.icon} viewBox="0 0 24 24" aria-hidden="true">
    {path}
  </svg>
);
const ICONS = {
  paper: icon(<path d="M6 3h8l4 4v14H6zM14 3v4h4M9 12h6M9 16h6" />),
  code: icon(<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M13.5 5l-3 14" />),
  dataset: icon(<path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />),
  bibtex: icon(<path d="M7 7h4v4c0 3-1.5 5-4 6M15 7h4v4c0 3-1.5 5-4 6" />),
};

// The dataset is the paper's main artifact, so its button is the filled one.
export default function LinkButtons() {
  return (
    <nav className={styles.buttons} aria-label="Paper, code, dataset and citation">
      <a className={styles.button} href={links.paper.href} target="_blank" rel="noopener noreferrer">
        {ICONS.paper}
        {links.paper.label}
      </a>
      <a className={`${styles.button} ${styles.primary}`} href={links.dataset.href} target="_blank" rel="noopener noreferrer">
        {ICONS.dataset}
        {links.dataset.label}
      </a>
      <a className={styles.button} href={links.code.href} target="_blank" rel="noopener noreferrer">
        {ICONS.code}
        {links.code.label}
      </a>
      <a className={styles.button} href={links.bibtex.href}>
        {ICONS.bibtex}
        {links.bibtex.label}
      </a>
    </nav>
  );
}
