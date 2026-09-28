"use client";

import type { CSSProperties, ReactNode } from "react";
import { cultureFlow } from "@/content";
import { useInView } from "./useInView";
import styles from "./CultureFlow.module.css";

// From flask to label, drawn in line art. A dot travels the track and each stage's ring lights
// as it passes; it runs only while on screen. Without motion it is a complete static diagram.
const X = [80, 240, 400, 560, 720, 880];
const Y = 50; // the track; the traveller's path in the CSS runs along it

const flask = (dx: number) => (
  <g transform={`translate(${dx} 0) scale(0.62)`}>
    <path d="M-5 -22v12l-13 22a3 3 0 0 0 3 4.5h30a3 3 0 0 0 3-4.5l-13-22v-12z" />
    <path className={styles.liquid} d="M-12.5 6h25l3.8 6.2a3 3 0 0 1-3 4.3h-26.6a3 3 0 0 1-3-4.3z" />
  </g>
);

const ICONS: ReactNode[] = [
  <g key="grow">{flask(-15)}{flask(0)}{flask(15)}</g>,
  <g key="verify">
    <circle cx="-4" cy="-4" r="13" />
    <path d="M5.5 5.5l11 11" />
    <path className={styles.accent} d="M-10 -4l4 4 8-8" />
  </g>,
  <g key="mix">
    <path d="M-9 -24v36a9 9 0 0 0 18 0v-36" />
    <path d="M-12 -24h24" />
    <circle className={styles.dotA} cx="-3" cy="2" r="2.4" />
    <circle className={styles.dotB} cx="3.5" cy="8" r="2.4" />
    <circle className={styles.dotA} cx="-2" cy="13" r="2.4" />
    <circle className={styles.dotB} cx="2.5" cy="-4" r="2.4" />
  </g>,
  <g key="mount">
    <rect x="-26" y="-8" width="52" height="16" rx="2.5" />
    <rect className={styles.cover} x="-10" y="-11" width="20" height="22" rx="1.5" />
  </g>,
  <g key="image">
    <path d="M-16 24h32M-2 24v-6M-14 10h26M9 24c0-14 2-24-8-34" />
    <path d="M-10 -20l9 -8 6 7-9 8z" />
    <path d="M-4 -12l6 14" />
  </g>,
  <g key="label">
    <rect x="-18" y="-24" width="36" height="28" rx="3" />
    <rect className={styles.rod} x="-12" y="-17" width="10" height="3.6" rx="1.8" transform="rotate(-20 -7 -15)" />
    <rect className={styles.rod} x="0" y="-10" width="12" height="3.6" rx="1.8" transform="rotate(15 6 -8)" />
    <rect className={styles.rod} x="-9" y="-6" width="6" height="3.6" rx="1.8" />
    <text x="-20" y="18">bs</text>
    <text x="-6" y="18">fj</text>
    <text x="7" y="18">ka</text>
  </g>,
];

export default function CultureFlow() {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <div ref={ref} className={`${styles.flow} ${inView ? styles.play : ""}`} data-flow="culture">
      {/* on phones the diagram keeps a readable size and scrolls sideways in its own box */}
      <div className={styles.scroller}>
        <svg className={styles.diagram} viewBox="0 0 960 100" role="img" aria-label="Six stages from separate cultures to a labelled image">
          <line className={styles.track} x1={X[0]} y1={Y} x2={X[5]} y2={Y} />
          <circle className={styles.traveller} r="5" />
          {cultureFlow.map((s, i) => (
            <g key={s.title} data-stage="" transform={`translate(${X[i]} ${Y})`}>
              <circle className={styles.ring} r="42" style={{ "--i": i } as CSSProperties} />
              <g className={styles.icon}>{ICONS[i]}</g>
            </g>
          ))}
        </svg>
      </div>
      <ol className={styles.captions}>
        {cultureFlow.map((s, i) => (
          <li key={s.title}>
            <span className={styles.step}>{String(i + 1).padStart(2, "0")}</span>
            <h3>{s.title}</h3>
            <p>{s.text}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
