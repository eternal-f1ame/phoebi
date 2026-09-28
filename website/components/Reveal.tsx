"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { inViewOptions, prefersReducedMotion } from "./useInView";
import styles from "./Reveal.module.css";

// Fades a section in as it scrolls into view. Sections already on screen at load, readers
// without JavaScript and reduced-motion visitors get it in place, without the effect, and print
// shows every section.
export default function Reveal({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"still" | "waiting" | "shown">("still");

  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion() || typeof IntersectionObserver === "undefined") return;
    if (el.getBoundingClientRect().top < window.innerHeight) return;
    setState("waiting");
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setState("shown");
        io.disconnect();
      }
    }, inViewOptions());
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} className={state === "still" ? undefined : styles[state]}>
      {children}
    </div>
  );
}
