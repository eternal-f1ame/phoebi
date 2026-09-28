"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { methodExample, methodFlow, species } from "@/content";
import { inViewOptions, prefersReducedMotion } from "./useInView";
import styles from "./DecoderFlow.module.css";

// How the decoders read one image, as a schematic that loops while it is in view: tiles, a
// frozen encoder, comparison with species prototypes, averaged scores, a decision.
// It renders empty ("armed"), so a diagram already on screen at load builds up without first
// flashing complete. The CSS shows that empty state only on screen, to a scripted page and a
// reader who allows motion, so print, reduced motion and no JavaScript get the finished diagram.
const CX = [95, 290, 480, 680, 870];
const CY = 104;
const CODES = species.map((s) => s.code);
const PRESENT = new Set(methodExample.present);
const HEIGHTS: Record<string, number> = { bs: 70, bt: 18, fj: 62, ka: 76, mx: 22, pf: 14 }; // schematic
const THRESHOLD = 44;
const PROTO = CODES.map((c, i) => {
  const a = (-90 + i * 60) * (Math.PI / 180);
  return { code: c, x: CX[2] + 58 * Math.cos(a), y: CY + 58 * Math.sin(a) };
});

// One run: the 4.8 s build and a hold on the finished diagram, then its moving parts fade out and,
// after a moment empty, the next run starts.
const HOLD = 9000; // ms from the start of a run to its fade
const FADE = 500;
const GAP = 250;

export default function DecoderFlow() {
  const ref = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"still" | "armed" | "play" | "fade">("armed");

  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
      setState("still");
      return;
    }
    // It starts once its top is in the upper two thirds of the window and loops while any of it
    // is on screen. Wholly off screen it stops and re-arms, so each visit starts from the top.
    let timers: number[] = [];
    let running = false;
    const stop = () => {
      timers.forEach((t) => window.clearTimeout(t));
      timers = [];
      running = false;
    };
    const run = () => {
      setState("play");
      timers = [
        window.setTimeout(() => setState("fade"), HOLD),
        window.setTimeout(() => setState("armed"), HOLD + FADE),
        window.setTimeout(run, HOLD + FADE + GAP),
      ];
    };
    const enter = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !running) {
        running = true;
        run();
      }
    }, inViewOptions("35%"));
    const leave = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) {
        stop();
        setState("armed");
      }
    });
    enter.observe(el);
    leave.observe(el);
    return () => {
      stop();
      enter.disconnect();
      leave.disconnect();
    };
  }, []);

  const grid = [1, 2, 3].flatMap((k) => [
    <line key={`v${k}`} x1={25 + k * 35} y1={34} x2={25 + k * 35} y2={174} />,
    <line key={`h${k}`} x1={25} y1={34 + k * 35} x2={165} y2={34 + k * 35} />,
  ]);

  return (
    <div ref={ref} className={`${styles.flow} ${state === "still" ? "" : styles[state]}`} data-flow="method">
      {/* on phones the diagram keeps a readable size and scrolls sideways in its own box */}
      <div className={styles.scroller}>
        <svg className={styles.diagram} viewBox="0 22 960 166" role="img" aria-label="Schematic of how the decoders read an image">
          <defs>
            <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0 0L8 4 0 8z" fill="#9b98a6" />
            </marker>
          </defs>
          {[[178, 222], [360, 410], [548, 612], [744, 800]].map(([a, b]) => (
            <line key={a} className={styles.arrow} x1={a} y1={CY} x2={b} y2={CY} markerEnd="url(#arrow)" />
          ))}

          {/* 1. tile */}
          <g data-stage="">
            <image href={methodExample.image} x="25" y="34" width="140" height="140" preserveAspectRatio="xMidYMid slice" />
            <rect className={styles.frame} x="25" y="34" width="140" height="140" />
            <g className={styles.grid}>{grid}</g>
          </g>

          {/* 2. encode */}
          <g data-stage="">
            {[0, 1, 2].map((k) => (
              <rect key={k} className={styles.tile} x={226} y={CY - 26 + k * 18} width="14" height="14" rx="2" style={{ "--k": k } as CSSProperties} />
            ))}
            <rect className={styles.box} x="250" y={CY - 44} width="100" height="88" rx="10" />
            <text className={styles.boxTitle} x="300" y={CY - 6}>Frozen</text>
            <text className={styles.boxText} x="300" y={CY + 13}>DINOv2-S/14</text>
          </g>

          {/* 3. compare with prototypes */}
          <g data-stage="">
            <circle className={styles.orbit} cx={CX[2]} cy={CY} r="58" />
            {PROTO.map((p) => (
              <g key={p.code}>
                <rect className={PRESENT.has(p.code) ? styles.protoOn : styles.proto} x={p.x - 5} y={p.y - 5} width="10" height="10" transform={`rotate(45 ${p.x} ${p.y})`} />
                <text className={styles.protoLabel} x={p.x + (p.x < CX[2] - 1 ? -12 : p.x > CX[2] + 1 ? 12 : 0)} y={p.y + (Math.abs(p.x - CX[2]) < 1 ? (p.y < CY ? -10 : 18) : 4)}>{p.code}</text>
              </g>
            ))}
            {PROTO.filter((p) => PRESENT.has(p.code)).map((p, k) => (
              <circle
                key={p.code}
                className={styles.point}
                cx={CX[2] + (p.x - CX[2]) * 0.72}
                cy={CY + (p.y - CY) * 0.72}
                r="4"
                style={{ "--dx": `${(CX[2] - p.x) * 0.72}px`, "--dy": `${(CY - p.y) * 0.72}px`, "--k": k } as CSSProperties}
              />
            ))}
          </g>

          {/* 4. average scores */}
          <g data-stage="">
            <line className={styles.threshold} x1="622" y1={160 - THRESHOLD} x2="742" y2={160 - THRESHOLD} />
            {CODES.map((c, k) => (
              <g key={c}>
                <rect className={PRESENT.has(c) ? styles.barOn : styles.bar} x={626 + k * 19} y={160 - HEIGHTS[c]} width="12" height={HEIGHTS[c]} rx="2" style={{ "--k": k } as CSSProperties} />
                <text className={styles.barLabel} x={632 + k * 19} y="176">{c}</text>
              </g>
            ))}
          </g>

          {/* 5. decide */}
          <g data-stage="">
            {CODES.map((c, k) => {
              const x = 818 + (k % 2) * 54;
              const y = 58 + Math.floor(k / 2) * 34;
              const on = PRESENT.has(c);
              return (
                <g key={c} className={on ? styles.chipOn : styles.chip} style={{ "--k": k } as CSSProperties}>
                  <rect x={x} y={y} width="46" height="24" rx="12" />
                  <text x={x + 23} y={y + 16}>{c}</text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <ol className={styles.captions}>
        {methodFlow.map((s, i) => (
          <li key={s.title}>
            <span className={styles.step}>{String(i + 1).padStart(2, "0")}</span>
            <h3>{s.title}</h3>
            <p>{s.text}</p>
          </li>
        ))}
      </ol>
      <p className={styles.note}>{methodExample.note}</p>
    </div>
  );
}
