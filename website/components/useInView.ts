"use client";

import { useEffect, useRef, useState } from "react";

export const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// "In view" means any part of the element is inside the window less its bottom `margin`. A ratio
// threshold would never be met by an element several times taller than the window, which is
// every section on a phone at 400% zoom.
export const inViewOptions = (margin = "10%"): IntersectionObserverInit => ({
  threshold: 0,
  rootMargin: `0px 0px -${margin} 0px`,
});

// True while the element is in view (or once it has been, with `once`). Without
// IntersectionObserver it reports true, so nothing waits for an event that never comes.
export function useInView<T extends Element>({ once = false } = {}) {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setInView(true);
        if (once) io.disconnect();
      } else if (!once) {
        setInView(false);
      }
    }, inViewOptions());
    io.observe(el);
    return () => io.disconnect();
  }, [once]);
  return [ref, inView] as const;
}
