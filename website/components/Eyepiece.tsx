import styles from "./Eyepiece.module.css";

type Props = {
  src: string;
  alt: string;
  width: number;
  height: number;
  size?: "large" | "small";
  reticle?: boolean;
  focus?: boolean; // focuses in on load, then refocuses every 10 s; off under prefers-reduced-motion
  lazy?: boolean;
};

// A micrograph as seen down a microscope: a round field in a dark ring, with an optional
// unlabelled reticle (the pixel size is not calibrated, so there is no scale bar).
const TICKS = Array.from({ length: 17 }, (_, i) => 10 + i * 5);

export default function Eyepiece({ src, alt, width, height, size = "large", reticle = true, focus = false, lazy = false }: Props) {
  return (
    <div className={`${styles.eyepiece} ${styles[size]}`}>
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        loading={lazy ? "lazy" : undefined}
        className={focus ? styles.focus : undefined}
        {...(focus ? { "data-focus": "" } : {})}
      />
      {reticle && (
        <svg className={styles.reticle} viewBox="0 0 100 100" aria-hidden="true">
          <line x1="0" y1="50" x2="100" y2="50" />
          <line x1="50" y1="0" x2="50" y2="100" />
          {TICKS.map((x) => (
            <line key={x} x1={x} y1={x % 25 === 0 ? 47.6 : 48.6} x2={x} y2={x % 25 === 0 ? 52.4 : 51.4} />
          ))}
        </svg>
      )}
    </div>
  );
}
