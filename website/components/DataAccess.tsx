import { dataAccess, sections, snippets, type Note } from "@/content";
import SectionHeading from "./SectionHeading";
import styles from "./DataAccess.module.css";

function NoteLine({ note }: { note: Note }) {
  return (
    <p className={styles.note}>
      {note.before}{" "}
      <a href={note.link.href} target="_blank" rel="noopener noreferrer">
        {note.link.label}
      </a>
      {note.after.startsWith(".") ? note.after : ` ${note.after}`}
    </p>
  );
}

export default function DataAccess() {
  const blocks = [
    { ...dataAccess.load, code: snippets.load, label: "Python code for loading the dataset" },
    { ...dataAccess.fetch, code: snippets.fetch, label: "Shell commands for downloading the code and the images" },
  ];
  return (
    <section id="data" className="section">
      <div className="container">
        <SectionHeading {...sections.data} />
        <div className={styles.blocks}>
          {blocks.map((b) => (
            <div key={b.title}>
              <h3>{b.title}</h3>
              {/* focusable so the block can be scrolled with the keyboard in every browser */}
              <pre className={styles.code} tabIndex={0} aria-label={b.label}>
                <code>{b.code}</code>
              </pre>
              <NoteLine note={b.note} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
