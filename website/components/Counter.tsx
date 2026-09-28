"use client";

import { useEffect, useRef, useState } from "react";
import { prefersReducedMotion, useInView } from "./useInView";
import styles from "./Counter.module.css";

// A number that counts up when it scrolls into view. The server renders the final value, so
// crawlers, readers without JavaScript and reduced-motion visitors always see the real number;
// screen readers and print get the real value from a second copy.
export default function Counter({ value }: { value: string }) {
  const m = value.match(/^([\d,]+)(.*)$/);
  const target = m ? Number(m[1].replace(/,/g, "")) : NaN;
  const commas = m ? m[1].includes(",") : false;
  const suffix = m ? m[2] : "";
  const format = (n: number) => (commas ? n.toLocaleString("en-US") : String(n)) + suffix;

  const [ref, inView] = useInView<HTMLSpanElement>({ once: true });
  const [shown, setShown] = useState(value);
  const armed = useRef(false);

  // Start from zero only if the number is still off screen, so nobody sees it jump back.
  useEffect(() => {
    const el = ref.current;
    if (!el || !Number.isFinite(target) || prefersReducedMotion()) return;
    if (el.getBoundingClientRect().top > window.innerHeight) {
      armed.current = true;
      setShown(format(0));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!inView || !armed.current) return;
    armed.current = false;
    const t0 = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const p = Math.min(1, (t - t0) / 1400);
      setShown(format(Math.round(target * (1 - Math.pow(1 - p, 3)))));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView]);

  return (
    <span ref={ref}>
      <span className={styles.counting} aria-hidden="true">
        {shown}
      </span>
      <span className={styles.final}>{value}</span>
    </span>
  );
}
