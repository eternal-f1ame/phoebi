"use client";

import { useEffect, useState } from "react";
import { combinations, leads, sections, species, task } from "@/content";
import Eyepiece from "./Eyepiece";
import SectionHeading from "./SectionHeading";
import styles from "./TryTask.module.css";

// The benchmark's task, by hand: look at a field, say which species are present, check.
// The first field is fixed so the server and the browser render the same page; the rest of
// the order is shuffled after load.
const FIRST = Math.max(0, combinations.findIndex((c) => c.code === task.first));

function shuffled(first: number): number[] {
  const rest = combinations.map((_, i) => i).filter((i) => i !== first);
  for (let i = rest.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rest[i], rest[j]] = [rest[j], rest[i]];
  }
  return [first, ...rest];
}

export default function TryTask() {
  const [order, setOrder] = useState<number[]>(() => combinations.map((_, i) => (i + FIRST) % combinations.length));
  const [pos, setPos] = useState(0);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [checked, setChecked] = useState(false);
  const [empty, setEmpty] = useState(false); // Check answer pressed with nothing picked
  const [tally, setTally] = useState({ fields: 0, perfect: 0 });

  useEffect(() => setOrder(shuffled(FIRST)), []);

  const combo = combinations[order[pos]];
  const truth = new Set(combo.species);
  const found = combo.species.filter((c) => picked.has(c)).length;
  const wrong = [...picked].filter((c) => !truth.has(c)).length;

  function toggle(code: string) {
    if (checked) return;
    setEmpty(false);
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
  }

  // Check answer stays focusable once used (aria-disabled, not disabled), so keyboard focus
  // does not fall back to the top of the page.
  function check() {
    if (checked) return;
    if (picked.size === 0) {
      setEmpty(true);
      return;
    }
    setChecked(true);
    setTally((t) => ({ fields: t.fields + 1, perfect: t.perfect + (found === truth.size && wrong === 0 ? 1 : 0) }));
  }

  function next() {
    setPos((p) => (p + 1) % order.length);
    setPicked(new Set());
    setChecked(false);
    setEmpty(false);
  }

  const names = species.filter((s) => truth.has(s.code)).map((s) => s.name);

  return (
    <section id="task" className="section">
      <div className="container">
        <SectionHeading {...sections.task} />
        <p className="lead">{leads.task}</p>
        <div className={styles.layout}>
          <div className={styles.viewer} key={combo.code}>
            <Eyepiece src={combo.image} alt="A phase-contrast field from one of the 40 cultures" width={480} height={480} />
          </div>
          <div>
            <p className={styles.label}>{task.pick}</p>
            <div className={styles.toggles} role="group" aria-label="Species">
              {species.map((s) => {
                const on = picked.has(s.code);
                const verdict = !checked ? "" : truth.has(s.code) ? (on ? styles.hit : styles.missed) : on ? styles.wrong : "";
                // after checking, the verdict is spelled out, not left to colour
                const mark = !checked ? "" : truth.has(s.code) ? task.present : on ? task.absent : "";
                return (
                  <button
                    key={s.code}
                    type="button"
                    className={`${styles.toggle} ${on ? styles.on : ""} ${verdict}`}
                    aria-pressed={on}
                    aria-label={mark ? `${s.name}: ${mark}` : s.name}
                    onClick={() => toggle(s.code)}
                    disabled={checked}
                  >
                    <span className={styles.code}>{s.code}</span>
                    <em>{s.name}</em>
                    {mark && <span className={`${styles.mark} ${truth.has(s.code) ? "" : styles.markAbsent}`}>{mark}</span>}
                  </button>
                );
              })}
            </div>
            <div className={styles.actions}>
              <button type="button" className={styles.primary} onClick={check} aria-disabled={checked}>
                {task.check}
              </button>
              <button type="button" className={styles.secondary} onClick={next}>
                {task.next}
              </button>
              {tally.fields > 0 && <span className={styles.tally}>{task.tally(tally.perfect, tally.fields)}</span>}
            </div>
            <div className={styles.result} aria-live="polite">
              {empty && !checked && <p className={styles.note}>{task.pickFirst}</p>}
              {checked && (
                <>
                  <p className={styles.score}>{task.score(found, truth.size, wrong)}</p>
                  <p>
                    {task.contains} <em>{names.join(", ")}</em>. {combo.heldOut ? task.heldOut : task.trained}
                  </p>
                  <p className={styles.note}>{task.note}</p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
