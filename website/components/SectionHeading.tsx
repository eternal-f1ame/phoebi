import styles from "./SectionHeading.module.css";

// The numbered heading that opens each section, like a paper's section title.
export default function SectionHeading({ number, title }: { number: string; title: string }) {
  return (
    <div className={styles.heading}>
      <span className={styles.number} aria-hidden="true">
        {number}
      </span>
      <h2>{title}</h2>
    </div>
  );
}
