// Shared helpers for the talk animations. Each page draws one SVG on a white ground (white survives
// H.264 exactly, so a video sits seamlessly on a white slide) and exposes render(t): a frame is a
// pure function of time, which render.mjs captures frame by frame.
const C = {
  ink: "#1c1b22",
  ink2: "#4a4854",
  muted: "#8c8996",
  rule: "#d8d2c5",
  surface: "#f7f5ef",
  violet: "#4b2e83",
  violetSoft: "#e9e3f3",
  safranin: "#b83b5e",
  safraninSoft: "#f6e3e9",
  grey: "#b9b6c1",
  white: "#ffffff",
};
const SPECIES = [
  { code: "bs", name: "B. subtilis" },
  { code: "bt", name: "B. thermoamylovorans" },
  { code: "fj", name: "F. johnsoniae" },
  { code: "ka", name: "K. aerogenes" },
  { code: "mx", name: "M. xanthus" },
  { code: "pf", name: "P. fluorescens" },
];
const NS = "http://www.w3.org/2000/svg";
const FONT = "Inter, 'Liberation Sans', sans-serif";
const MONO = "'DejaVu Sans Mono', monospace";

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, p) => a + (b - a) * p;
const ease = {
  linear: (t) => t,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t) => 1 - Math.pow(1 - t, 3),
  in: (t) => t * t * t,
  back: (t) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
};
// progress of t through [a, b], eased
const seg = (t, a, b, e = ease.inOut) => e(clamp((t - a) / (b - a)));

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

function txt(parent, x, y, s, o = {}) {
  const t = el("text", {
    x, y,
    "font-family": o.mono ? MONO : FONT,
    "font-size": o.size || 22,
    "font-weight": o.weight || 400,
    "font-style": o.italic ? "italic" : "normal",
    fill: o.fill || C.ink,
    "text-anchor": o.anchor || "start",
    "dominant-baseline": o.baseline || "alphabetic",
    "letter-spacing": o.spacing || 0,
  }, parent);
  t.textContent = s;
  return t;
}

// A group placed at (x, y), scaled about its own origin.
function place(g, x, y, s = 1, r = 0) {
  g.setAttribute("transform", `translate(${x} ${y}) rotate(${r}) scale(${s})`);
}

function show(e, o) {
  e.setAttribute("opacity", clamp(o));
}

function setup(W, H, DUR, HOLD = 0) {
  Object.assign(window, { W, H, DUR, HOLD });
  document.body.style.cssText = `margin:0;background:#fff;width:${W}px;height:${H}px;overflow:hidden`;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  document.body.appendChild(svg);
  el("rect", { width: W, height: H, fill: C.white }, svg);
  return svg;
}

// Pages call ready() once their images have decoded, so the first captured frame is complete.
async function ready() {
  await document.fonts.ready;
  const imgs = [...document.querySelectorAll("image")];
  await Promise.all(
    imgs.map((im) => new Promise((res) => {
      const probe = new Image();
      probe.onload = probe.onerror = res;
      probe.src = im.getAttribute("href");
    })),
  );
  window.render(0);
  window.READY = true;
}

// A rounded chip with a centred label, drawn about its centre; returns its parts for restyling.
function chip(parent, cx, cy, w, h, label, o = {}) {
  const g = el("g", {}, parent);
  const r = el("rect", { x: -w / 2, y: -h / 2, width: w, height: h, rx: h / 2, fill: C.white, stroke: C.rule, "stroke-width": 2 }, g);
  const t = txt(g, 0, 1, label, { size: o.size || 22, anchor: "middle", baseline: "middle", mono: o.mono !== false, weight: o.weight || 600, fill: C.muted });
  place(g, cx, cy);
  return { g, r, t, cx, cy };
}

function styleChip(c, on, color = C.violet) {
  c.r.setAttribute("fill", on ? color : C.white);
  c.r.setAttribute("stroke", on ? color : C.rule);
  c.t.setAttribute("fill", on ? C.white : C.muted);
}

// Mix two hex colours.
function mix(a, b, p) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, i) => Math.round(lerp(v, pb[i], clamp(p))).toString(16).padStart(2, "0")).join("");
}
