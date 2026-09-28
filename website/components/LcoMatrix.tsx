"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { combinations, species, splitSwitch, type Combination } from "@/content";
import Eyepiece from "./Eyepiece";
import styles from "./LcoMatrix.module.css";

// The 40 combinations as columns (rows are species), switchable between the random split and
// leave-combinations-out. Hover, tap or focus a column to see that culture. The columns are one
// tab stop; the arrow keys, Home and End move between them.
const CELL = 13;
const STEP = 16; // column pitch
const GROUP_GAP = 18;
const LEFT = 30;
const TOP = 34;
const ORDERS = [1, 2, 3, 4, 6];

const columns = (() => {
  let x = LEFT;
  let prev = 0;
  return combinations.map((c) => {
    const order = c.species.length;
    if (prev && order !== prev) x += GROUP_GAP;
    prev = order;
    const col = { combo: c, x };
    x += STEP;
    return col;
  });
})();
const WIDTH = columns[columns.length - 1].x + CELL / 2 + 30; // room for the last group's label
const HEIGHT = TOP + species.length * STEP + 8;

type Mode = "random" | "lco";

export default function LcoMatrix() {
  const [mode, setMode] = useState<Mode>("random");
  const [active, setActive] = useState<Combination | null>(null);
  const [stop, setStop] = useState(0); // the column that takes the tab stop
  const refs = useRef<(SVGGElement | null)[]>([]);
  const names = (c: Combination) => species.filter((s) => c.species.includes(s.code)).map((s) => s.name);

  function onKey(e: KeyboardEvent, i: number) {
    const last = columns.length - 1;
    const to = ({ ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: last } as Record<string, number>)[e.key];
    if (to !== undefined) {
      e.preventDefault();
      refs.current[Math.max(0, Math.min(last, to))]?.focus();
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setActive(columns[i].combo);
    }
  }
  const role = (c: Combination) =>
    mode === "random"
      ? "Seen in training: its images are split 80/10/10."
      : c.heldOut
        ? "Held out: never seen in training."
        : "Used for training and validation.";

  return (
    <div className={`${styles.matrix} ${mode === "lco" ? styles.lco : ""}`}>
      {/* the switch, legend and matrix on the left; the panel on the right runs their full height */}
      <div className={styles.main}>
        <div className={styles.switch} role="group" aria-label="Split">
          {(["random", "lco"] as Mode[]).map((m) => (
            <button key={m} type="button" aria-pressed={mode === m} className={mode === m ? styles.active : ""} onClick={() => setMode(m)}>
              {splitSwitch[m].label}
            </button>
          ))}
        </div>
        <p className={styles.legend}>{splitSwitch[mode].legend}</p>
        {/* on phones the matrix keeps tappable columns and scrolls sideways in its own box */}
        <div className={styles.scroller}>
          <svg className={styles.grid} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="toolbar" aria-label="The 40 culture combinations">
            {ORDERS.map((o, k) => {
              const cols = columns.filter((c) => c.combo.species.length === o);
              const mid = (cols[0].x + cols[cols.length - 1].x + CELL) / 2;
              return (
                <text key={o} className={styles.groupLabel} x={mid} y={12}>
                  {splitSwitch.orders[k]}
                </text>
              );
            })}
            {species.map((s, r) => (
              <text key={s.code} className={styles.rowLabel} x={LEFT - 8} y={TOP + r * STEP + 10}>
                {s.code}
              </text>
            ))}
            {columns.map(({ combo, x }, i) => {
              const state = mode === "random" ? "random" : combo.heldOut ? "held-out" : "train";
              const label = `${names(combo).join(", ")}${mode === "lco" && combo.heldOut ? ", held out" : ""}`;
              return (
                <g
                  key={combo.code}
                  ref={(el) => {
                    refs.current[i] = el;
                  }}
                  data-combo={combo.code}
                  data-state={state}
                  className={`${styles.column} ${state === "held-out" ? styles.heldOut : ""} ${active?.code === combo.code ? styles.current : ""}`}
                  tabIndex={i === stop ? 0 : -1}
                  role="button"
                  aria-label={label}
                  onMouseEnter={() => setActive(combo)}
                  onFocus={() => {
                    setStop(i);
                    setActive(combo);
                  }}
                  onKeyDown={(e) => onKey(e, i)}
                  onClick={() => setActive(combo)}
                >
                  <rect className={styles.band} x={x - 2} y={TOP - 4} width={CELL + 4} height={species.length * STEP + 4} rx="3" />
                  {species.map((s, r) => (
                    <rect
                      key={s.code}
                      className={combo.species.includes(s.code) ? styles.present : styles.absent}
                      x={x}
                      y={TOP + r * STEP}
                      width={CELL}
                      height={CELL}
                      rx="2"
                    />
                  ))}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* not a live region: each column's own label already names its species and role */}
      <aside className={styles.info}>
        {active ? (
          <>
            <div className={styles.thumb}>
              <Eyepiece src={active.image} alt={`A phase-contrast field from the ${active.code} culture`} width={480} height={480} size="small" reticle={false} />
            </div>
            <p className={styles.combo}>{active.code}</p>
            <p className={styles.species}>
              <em>{names(active).join(", ")}</em>
            </p>
            <p className={styles.role}>{role(active)}</p>
          </>
        ) : (
          <p className={styles.hint}>{splitSwitch.hint}</p>
        )}
      </aside>
    </div>
  );
}
