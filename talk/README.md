# Talk animations

The animated figures for the five-minute NeurIPS 2026 talk. Each page in `anim/` draws one SVG and
exposes `render(t)`, so a frame is a pure function of time; `render.mjs` captures the pages frame by
frame with Playwright and encodes them as H.264 MP4 at 30 fps, each ending on its finished frame.
The backgrounds are pure white, which survives the encoding exactly, so the videos sit seamlessly
on a white slide.

| Video | Slide | Shows |
|---|---|---|
| `01-eyepiece` | Title | the six-species field coming into focus; the six species light up |
| `02-species` | The problem | the six species as look-alike pairs, and their cell lengths (Table 2) |
| `03-culture` | The benchmark | grow, verify, mix, mount, image, label |
| `04-lco` | The question | the 40 combinations; leave-combinations-out lifts out the nine held out |
| `05-collapse` | Finding | in-distribution to held-out F1 for every supervised baseline (Table 3) |
| `06-joint` | Diagnosis | schematic: a per-image head answers with a seen mixture; anchors read each species |
| `07-decoder` | Method | Assumption H, then tile, encode, compare, average, decide |
| `08-result` | Result | the same chart with the three anchor decoders added (Table 4) |
| `09-openworld` | Open world | schematic of the held-out-`ka` fold: residual misses it, k-NN flags it, Sinkhorn-Knopp names it |

Numbers in `05` and `08` are the paper's; `06`, `07` and `09` are schematics and say so on screen.

```bash
node talk/render.mjs talk/anim/05-collapse.html                 # one video into talk/out/
node talk/render.mjs --stills 2,6,11 talk/anim/05-collapse.html # PNG stills into talk/out/stills/
```

`render.mjs` imports Playwright from `/opt/node-tools/node_modules/playwright`; point the import at
your own install elsewhere. The pages use Inter, falling back to Liberation Sans.
