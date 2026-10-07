// Captures each animation page frame by frame and encodes it as H.264 MP4 (plus a poster of the
// last frame). Usage:
//   node talk/render.mjs talk/anim/05-collapse.html [...]         full videos into talk/out/
//   node talk/render.mjs --stills 0,3,8 talk/anim/05-collapse.html  PNG stills into talk/out/stills/
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "/opt/node-tools/node_modules/playwright/index.mjs";

const FPS = 30;
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, "out");
const args = process.argv.slice(2);
let stills = null;
if (args[0] === "--stills") {
  stills = args[1].split(",").map(Number);
  args.splice(0, 2);
}

const browser = await chromium.launch();
for (const file of args) {
  const name = path.basename(file, ".html");
  const page = await browser.newPage({ viewport: { width: 64, height: 64 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.error(`${name}: ${e.message}`));
  await page.goto("file://" + path.resolve(file));
  await page.waitForFunction(() => window.READY === true);
  const { W, H, DUR, HOLD } = await page.evaluate(() => ({ W: window.W, H: window.H, DUR: window.DUR, HOLD: window.HOLD }));
  await page.setViewportSize({ width: W, height: H });

  if (stills) {
    mkdirSync(path.join(outDir, "stills"), { recursive: true });
    for (const t of stills) {
      await page.evaluate((t) => window.render(t), t);
      await page.screenshot({ path: path.join(outDir, "stills", `${name}-${t}.png`) });
    }
    console.log(`${name}: stills at ${stills.join(", ")} s`);
    await page.close();
    continue;
  }

  const out = path.join(outDir, `${name}.mp4`);
  const ff = spawn("ffmpeg", [
    "-y", "-loglevel", "error",
    "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
    "-vf", `tpad=stop_mode=clone:stop_duration=${HOLD}`,
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-movflags", "+faststart", out,
  ], { stdio: ["pipe", "inherit", "inherit"] });
  const n = Math.round(DUR * FPS) + 1;
  for (let i = 0; i < n; i++) {
    await page.evaluate((t) => window.render(t), i / FPS);
    const buf = await page.screenshot({ type: "png" });
    if (!ff.stdin.write(buf)) await once(ff.stdin, "drain");
  }
  ff.stdin.end();
  await once(ff, "close");
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  console.log(`${name}: ${W}x${H}, ${DUR}s build + ${HOLD}s hold -> ${path.relative(process.cwd(), out)}`);
  await page.close();
}
await browser.close();
