// Check the static export in out/ for everything the page must carry.
//
//   npm run build && npm run check            content, anchors, local files, page weight
//   npm run check -- --links                  also request every external link
//
// Exits 1 and lists every miss; the checks read the rendered text, not the source.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = fileURLToPath(new URL("../out/", import.meta.url));
const MAX_BYTES = 2e6;
const TITLE = "PHOEBI: An Open-World Benchmark for Multi-Label Bacterial Identification in Phase-Contrast Microscopy";
const AUTHORS = ["Aaditya Baranwal", "Md Jahid Hasan", "Shruti Vyas"];
const SPECIES = [
  "Bacillus subtilis", "Bacillus thermoamylovorans", "Flavobacterium johnsoniae",
  "Klebsiella aerogenes", "Myxococcus xanthus", "Pseudomonas fluorescens",
];
const SECTIONS = ["task", "species", "collection", "method", "benchmark", "results", "findings", "data", "citation"];
const CODE = "https://github.com/eternal-f1ame/phoebi";
const DATASET = "https://huggingface.co/datasets/sochastic/PHOEBI";
const NEURIPS = "https://neurips.cc/Conferences/2026";

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };
const decode = (s) => s
  .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
  .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
  .replace(/&([a-z]+);/gi, (m, n) => ENTITIES[n.toLowerCase()] ?? m);
const textOf = (html) => decode(html
  .replace(/<script[\s\S]*?<\/script>/gi, " ")
  .replace(/<style[\s\S]*?<\/style>/gi, " ")
  .replace(/<!--[\s\S]*?-->/g, "")
  .replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ");

const misses = [];
const need = (ok, what) => { if (!ok) misses.push(what); };

// Status of a URL. Bodies are discarded and requests time out, so no socket keeps the process alive.
async function probe(url) {
  try {
    const res = await fetch(url, { redirect: "follow", signal: AbortSignal.timeout(15000) });
    await res.body?.cancel();
    return { status: res.status, type: res.headers.get("content-type") ?? "" };
  } catch (e) {
    return { status: `error ${e.cause?.code ?? e.name}`, type: "" };
  }
}

const indexPath = join(OUT, "index.html");
if (!existsSync(indexPath)) {
  console.error("out/index.html missing: run `npm run build` first");
  process.exit(1);
}
const html = readFileSync(indexPath, "utf8");
const text = textOf(html);

// Inline tags split text nodes, which textOf turns into spaces a browser would not show.
const squash = (s) => s.replace(/\s+/g, "");
need(squash(text).includes(squash(TITLE)), `title "${TITLE}"`);
for (const a of AUTHORS) need(text.includes(a), `author ${a}`);
need(text.includes("Institute of Artificial Intelligence"), "affiliation: Institute of Artificial Intelligence");
need(text.includes("University of Central Florida"), "affiliation: University of Central Florida");
need(/coming soon/i.test(text), 'the Paper button\'s "coming soon"');
const anchors = [...html.matchAll(/<a\b[^>]*>([\s\S]*?)<\/a>/gi)].map((m) => textOf(m[1]));
need(!anchors.some((t) => /\bPaper\b/.test(t)), "Paper must not be a link yet");
need(html.includes(`href="${CODE}`), `a link to ${CODE}`);
need(html.includes(`href="${DATASET}`), `a link to ${DATASET}`);
need(html.includes(`href="${NEURIPS}"`), `a link to the NeurIPS 2026 website (${NEURIPS})`);

const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
for (const s of SECTIONS) need(ids.has(s), `section id "${s}"`);
for (const [, target] of html.matchAll(/href="#([^"]*)"/g)) need(ids.has(target), `anchor target #${target}`);

for (const s of SPECIES) need(text.includes(s), `species ${s}`);
need(text.includes("@inproceedings{baranwal2026phoebi"), "the BibTeX entry");
need(/<meta[^>]+property="og:image"/.test(html), "an og:image meta tag");
need(!/acknowledg/i.test(text), "no acknowledgments section");
need(!text.includes("Optical microscopy (OM) enables rapid"), "no paper abstract on the page");
need(/Check answer/.test(text), 'the "Try the task" check button');
// the protocol list names the same two splits, so look for the switch's own buttons
need(
  /<button[^>]*aria-pressed="true"[^>]*>Random split<\/button>/.test(html) &&
    /<button[^>]*aria-pressed="false"[^>]*>Leave-combinations-out<\/button>/.test(html),
  "the split switch, on the random split",
);

// Sizes in the stylesheets are relative (vw, vh, rem, em, %), so the page scales with the window.
// Allowed in px: 1-2 px hairlines, media-query breakpoints, and inside the SVG diagrams the label
// sizes and moves, which are in the drawing's own units and scale with it.
const SRC = fileURLToPath(new URL("../", import.meta.url));
const SVG_MODULES = ["CultureFlow.module.css", "DecoderFlow.module.css", "LcoMatrix.module.css"];
const styles = ["app/globals.css", ...readdirSync(join(SRC, "components")).filter((f) => f.endsWith(".css")).map((f) => `components/${f}`)];
for (const file of styles) {
  readFileSync(join(SRC, file), "utf8").split("\n").forEach((line, i) => {
    const px = [...line.matchAll(/(-?\d*\.?\d+)px\b/g)].map((m) => Math.abs(Number(m[1])));
    if (!px.length || line.includes("@media") || px.every((n) => n <= 2)) return;
    if (SVG_MODULES.some((m) => file.endsWith(m)) && /font-size|translate/.test(line)) return;
    need(false, `a px size in ${file}:${i + 1}: ${line.trim()}`);
  });
}

// Every same-site file the page references must exist; together they must stay light. That
// includes the images its scripts load later (the quiz fields and the matrix thumbnails).
const refs = new Set();
for (const [, url] of html.matchAll(/\b(?:src|href)="(\/[^"#?]*)/g)) if (!url.startsWith("//")) refs.add(url);
for (const [, set] of html.matchAll(/\bsrcSet="([^"]+)"/gi)) for (const part of set.split(",")) refs.add(part.trim().split(/\s+/)[0]);
const chunks = join(OUT, "_next/static/chunks");
for (const name of existsSync(chunks) ? readdirSync(chunks, { recursive: true }) : []) {
  if (!String(name).endsWith(".js")) continue;
  for (const [, url] of readFileSync(join(chunks, String(name)), "utf8").matchAll(/"(\/img\/[^"\\]+)"/g)) refs.add(url);
}
let bytes = statSync(indexPath).size;
for (const ref of refs) {
  if (ref === "/") continue;
  const file = join(OUT, decodeURIComponent(ref));
  if (existsSync(file) && statSync(file).isFile()) bytes += statSync(file).size;
  else need(false, `referenced file ${ref}`);
}
need(bytes < MAX_BYTES, `page weight ${(bytes / 1e6).toFixed(2)} MB (limit ${MAX_BYTES / 1e6} MB)`);

if (process.argv.includes("--links")) {
  // The link preview must resolve on the live site, or shared links show no picture.
  const ogImage = decode(html.match(/<meta[^>]+property="og:image"[^>]+content="([^"]+)"/)?.[1] ?? "");
  const og = ogImage ? await probe(ogImage) : { status: "missing", type: "" };
  console.log(`og:image ${ogImage} → ${og.status} ${og.type}`);
  need(og.status === 200 && og.type.startsWith("image/png"), `og:image ${ogImage} answered ${og.status} ${og.type}`);
  const external = [...new Set([...html.matchAll(/href="(https?:\/\/[^"]+)"/g)].map((m) => decode(m[1])))];
  for (const url of external) {
    const { status } = await probe(url);
    console.log(`${String(status).padEnd(6)} ${url}`);
    need(typeof status === "number" && status < 400, `link ${url} answered ${status}`);
  }
}

console.log(`page weight: ${(bytes / 1e3).toFixed(0)} kB over ${refs.size} local files`);
if (misses.length) {
  console.log(`FAIL: ${misses.length} missing\n  - ${misses.join("\n  - ")}`);
  process.exit(1);
}
console.log("PASS: all checks");
process.exit(0);
