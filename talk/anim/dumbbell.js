// In-distribution vs held-out F1 under leave-combinations-out, one arrow per model (paper Tables 3
// and 4). MODE "collapse": the supervised models fall. MODE "result": that state compresses left
// and the three anchor decoders arrive beside it.
const MODELS = [
  // [label, group, in-distribution F1, held-out F1]
  ["ResNet-50", "ft", 1.0, 0.509],
  ["ConvNeXt-B", "ft", 1.0, 0.606],
  ["ViT-B/16", "ft", 0.999, 0.536],
  ["DINOv2-S/14", "ft", 0.989, 0.437],
  ["DINOv3-S/16", "ft", 1.0, 0.505],
  ["CLIP B/16", "ft", 0.999, 0.435],
  ["SigLIP B/16", "ft", 0.997, 0.467],
  ["EVA-02 B/16", "ft", 0.999, 0.501],
  ["DINOv2, our tiles", "pipe", 1.0, 0.564],
  ["attention-MIL", "mil", 0.906, 0.574],
  ["SimplexUnmix", "ours", 0.579, 0.66],
  ["ProtoMatch", "ours", 0.59, 0.683],
  ["ChannelGroup", "ours", 0.689, 0.635],
];
const CONSTANT = 0.654; // the all-present predictor on held-out combinations

function dumbbell(MODE) {
  const W = 1180, H = 760;
  const svg = setup(W, H, MODE === "collapse" ? 11 : 10.5, 1);
  const X0 = 112, X1 = 1150, YT = 92, YB = 520; // plot box; F1 1.0 at YT, 0.4 at YB
  const fy = (f) => lerp(YB, YT, (f - 0.4) / 0.6);

  // column centres for a layout with or without our group
  function layout(withOurs) {
    const groups = withOurs ? ["ft", "pipe", "mil", "ours"] : ["ft", "pipe", "mil"];
    const cols = MODELS.filter((m) => groups.includes(m[1]));
    const gap = withOurs ? 1.0 : 1.1; // extra space between groups, in column units
    const units = cols.length - 1 + gap * (groups.length - 1);
    const pad = 46;
    const step = (X1 - X0 - 2 * pad) / units;
    const xs = {};
    let x = X0 + pad, prev = null;
    for (const m of cols) {
      if (prev && m[1] !== prev) x += gap * step;
      xs[m[0]] = x;
      x += step;
      prev = m[1];
    }
    return xs;
  }
  const L0 = layout(false), L1 = layout(true);

  // grid
  const grid = el("g", {}, svg);
  for (let f = 0.4; f <= 1.001; f += 0.1) {
    el("line", { x1: X0, x2: X1, y1: fy(f), y2: fy(f), stroke: C.rule, "stroke-width": 1.5, "stroke-dasharray": f > 0.41 ? "3 6" : "" }, grid);
    txt(grid, X0 - 16, fy(f) + 7, f.toFixed(1), { size: 20, anchor: "end", fill: C.muted });
  }
  txt(grid, X0 - 16, YT - 34, "per-image F1", { size: 20, fill: C.muted, weight: 500 });

  // legend
  const legend = el("g", {}, svg);
  el("circle", { cx: 250, cy: 28, r: 9, fill: C.white, stroke: C.ink2, "stroke-width": 3 }, legend);
  txt(legend, 268, 35, "mixtures seen in training", { size: 21, fill: C.ink2 });
  el("circle", { cx: 560, cy: 28, r: 9, fill: C.ink2 }, legend);
  txt(legend, 578, 35, "mixtures never seen", { size: 21, fill: C.ink2 });

  // the constant all-present predictor
  const constG = el("g", {}, svg);
  const constLine = el("line", { x1: X0, x2: X1, y1: fy(CONSTANT), y2: fy(CONSTANT), stroke: C.ink2, "stroke-width": 2.5, "stroke-dasharray": "10 8" }, constG);
  const constLabel = el("g", {}, svg);
  el("line", { x1: 812, x2: 852, y1: 28, y2: 28, stroke: C.ink2, "stroke-width": 2.5, "stroke-dasharray": "10 8" }, constLabel);
  txt(constLabel, 864, 35, "always “all six”: 0.654", { size: 21, fill: C.ink2, weight: 600 });

  // our band: where every fine-tuned model lands on unseen mixtures
  const band = el("g", {}, svg);
  const bandRect = el("rect", { x: X0, y: fy(0.606), width: X1 - X0, height: fy(0.435) - fy(0.606), fill: C.safraninSoft }, band);
  const bandLabel = txt(band, X1 - 12, fy(0.435) - 14, "every baseline, unseen: 0.44–0.61", { size: 20, anchor: "end", fill: C.safranin, weight: 600 });
  svg.insertBefore(band, grid.nextSibling);

  // one arrow per model
  const marks = MODELS.map((m, i) => {
    const color = m[1] === "ours" ? C.violet : C.safranin;
    const g = el("g", {}, svg);
    const line = el("line", { stroke: color, "stroke-width": 4, "stroke-linecap": "round" }, g);
    const head = el("path", { fill: color }, g);
    const end = el("circle", { r: 10, fill: color }, g);
    const start = el("circle", { r: 10, fill: C.white, stroke: color, "stroke-width": 3.5 }, g);
    const value = txt(g, 0, 0, m[3].toFixed(2), { size: 20, anchor: "middle", fill: color, weight: 700 });
    const label = txt(g, 0, 0, m[0], { size: 19, anchor: "end", fill: m[1] === "ours" ? C.violet : C.ink2, weight: m[1] === "ours" ? 700 : 500 });
    return { m, i, g, line, head, end, start, value, label };
  });

  // group brackets
  const brackets = [
    ["ft", "fine-tuned end to end"],
    ["pipe", "same|pipeline"],
    ["mil", "frozen|backbone"],
    ["ours", "ours: anchor|decoders"],
  ].map(([grp, name]) => {
    const g = el("g", {}, svg);
    const line = el("path", { fill: "none", stroke: grp === "ours" ? C.violet : C.rule, "stroke-width": 2.5 }, g);
    const t = txt(g, 0, 0, "", { size: 20, anchor: "middle", fill: grp === "ours" ? C.violet : C.ink2, weight: grp === "ours" ? 700 : 600 });
    name.split("|").forEach((part, i) => {
      const sp = el("tspan", { x: 0, dy: i ? 24 : 0 }, t);
      sp.textContent = part;
    });
    return { grp, g, line, t };
  });

  function render(t) {
    const comp = MODE === "result" ? seg(t, 0.3, 1.8) : 0; // layout morph
    const xOf = (name) => (name in L0 ? lerp(L0[name], L1[name], comp) : L1[name]);

    show(grid, MODE === "collapse" ? seg(t, 0, 0.9) : 1);
    show(legend, MODE === "collapse" ? seg(t, 0.8, 1.6) : 1);

    for (const k of marks) {
      const [name, grp, fin, fout] = k.m;
      const ours = grp === "ours";
      if (MODE === "collapse" && ours) { show(k.g, 0); continue; }
      const x = xOf(name);
      // timing: supervised build in collapse mode; ours build in result mode
      let pIn, pDrop;
      if (MODE === "collapse") {
        pIn = seg(t, 1.4 + k.i * 0.1, 1.9 + k.i * 0.1, ease.back);
        pDrop = seg(t, 3.2 + k.i * 0.22, 4.4 + k.i * 0.22, ease.inOut);
      } else if (ours) {
        const j = k.i - 10;
        pIn = seg(t, 2.0 + j * 0.25, 2.6 + j * 0.25, ease.back);
        pDrop = seg(t, 3.4 + j * 0.35, 4.6 + j * 0.35, ease.inOut);
      } else {
        pIn = 1;
        pDrop = 1;
      }
      const dim = MODE === "result" && !ours ? lerp(1, 0.38, seg(t, 0.4, 1.6)) : 1;
      show(k.g, pIn > 0 ? dim : 0);
      const y0 = fy(fin), y1 = fy(lerp(fin, fout, pDrop));
      k.start.setAttribute("cx", x); k.start.setAttribute("cy", y0);
      k.start.setAttribute("r", 10 * Math.max(0.01, pIn));
      const down = fout < fin ? 1 : -1;
      const yTip = y1 - down * 12;
      const showLine = Math.abs(y1 - y0) > 26;
      k.line.setAttribute("x1", x); k.line.setAttribute("y1", y0 + down * 12);
      k.line.setAttribute("x2", x); k.line.setAttribute("y2", showLine ? yTip - down * 8 : y0 + down * 12);
      k.head.setAttribute("d", showLine ? `M${x - 9} ${yTip - down * 12} L${x + 9} ${yTip - down * 12} L${x} ${yTip} Z` : "");
      k.end.setAttribute("cx", x); k.end.setAttribute("cy", y1);
      k.end.setAttribute("r", pDrop > 0.02 ? 10 : 0);
      // value labels: ours only, beside the held-out dot
      if (ours) {
        k.value.setAttribute("x", x); k.value.setAttribute("y", fout > fin ? fy(fout) - 22 : fy(fout) + 38);
        show(k.value, seg(t, 6.2, 6.9));
      } else show(k.value, 0);
      // x-axis label, rotated
      k.label.setAttribute("transform", `translate(${x + 6} ${YB + 26}) rotate(-35)`);
      k.label.setAttribute("x", 0); k.label.setAttribute("y", 0);
    }

    // brackets under each group
    for (const b of brackets) {
      const names = MODELS.filter((m) => m[1] === b.grp).map((m) => m[0]);
      if (MODE === "collapse" && b.grp === "ours") { show(b.g, 0); continue; }
      const xa = xOf(names[0]) - 24, xb = xOf(names[names.length - 1]) + 24;
      const y = 676;
      b.line.setAttribute("d", `M${xa} ${y - 10} V${y} H${xb} V${y - 10}`);
      const mid = (xa + xb) / 2;
      b.t.setAttribute("x", mid); b.t.setAttribute("y", y + 32);
      for (const sp of b.t.querySelectorAll("tspan")) sp.setAttribute("x", mid);
      if (MODE === "collapse") show(b.g, seg(t, 2.2, 3.0));
      else show(b.g, b.grp === "ours" ? seg(t, 2.0, 2.8) : lerp(1, 0.6, seg(t, 0.4, 1.6)));
    }

    // the constant predictor's line draws across after the fall
    const pc = MODE === "collapse" ? seg(t, 6.6, 7.8) : 1;
    constLine.setAttribute("x2", lerp(X0, X1, pc));
    show(constLabel, MODE === "collapse" ? seg(t, 6.6, 7.4) : 1);
    show(constG, 1);

    // the fine-tuned band: shown in the result, after ours arrive
    const pb = MODE === "result" ? seg(t, 5.4, 6.4) : MODE === "collapse" ? seg(t, 8.8, 9.8) : 0;
    bandRect.setAttribute("width", (X1 - X0) * pb);
    show(bandLabel, pb);
  }
  window.render = render;
  ready();
}
