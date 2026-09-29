import styles from "./SectionHeading.module.css";

// The heading that opens each section.
export default function SectionHeading({ title }: { title: string }) {
  return (
    <div className={styles.heading}>
      <h2>{title}</h2>
    </div>
  );
}
