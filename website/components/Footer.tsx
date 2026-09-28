import { footer } from "@/content";
import styles from "./Footer.module.css";

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`container ${styles.row}`}>
        <p>{footer.copyright}</p>
        <p>
          {footer.licences.map((l, i) => (
            <span key={l.label}>
              {i > 0 && " · "}
              {l.label}{" "}
              <a href={l.link.href} target="_blank" rel="noopener noreferrer">
                {l.link.label}
              </a>
            </span>
          ))}
        </p>
      </div>
    </footer>
  );
}
