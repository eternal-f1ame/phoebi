#!/usr/bin/env python3
"""Acquisition-shift robustness, illumination attribution, and a label-noise bound.

Three questions, one feature-extraction pass over the test split.

1. Does the finding survive an acquisition perturbation?

   A corruption battery is applied to the *full frame before tiling*, which is
   where a second microscope or session would actually differ: condenser
   misalignment (illum_gradient), objective/condenser vignetting, lamp
   intensity, detector gamma, focus drift, sensor noise, capture-software JPEG.
   The decoders are never retrained -- prototypes and thresholds are the frozen
   ones from the paper runs -- so this measures deployment-time robustness.

2. What does the illumination correction actually buy?

   Every condition is run twice, under --illumination divide (canonical,
   sigma=64) and none. On the in-distribution split the correction *costs* F1
   (outputs/ablations/illumination). If the ordering inverts under the
   illumination-family corruptions, that cost is the price of not keying on an
   acquisition-specific cue, and the decoders are not exploiting an illumination
   pattern.

3. How much label noise is there?

   Culture-derived labels can overstate presence: a species that is plated but
   outcompeted may be absent from a given field. This is boundable without any
   manual annotation, using the singleton combinations as a control.

     In a PURE culture there is nothing else in the dish, so the labelled
     species is present in every field by construction. A false negative there
     is model error and cannot be label noise.

     In a MIXTURE the same species can genuinely be missing from a field.

   So per species k, FNR_mixture(k) - FNR_singleton(k) is an estimate of the
   per-field rate at which the culture label overstates presence. It is an
   UPPER bound, not a point estimate: mixtures are also intrinsically harder
   (crowding, occlusion, fewer cells of each species per field), and that
   difficulty inflates the gap on top of any true label noise.

Outputs
-------
<output_dir>/results.json      every condition, every decoder
<output_dir>/results.csv       flat roll-up
<output_dir>/label_noise.json  per-species singleton-vs-mixture FNR
<output_dir>/summary.md        human-readable
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.io import load_json, load_real_split, save_json
from src.common.metrics import exact_match_accuracy, macro_f1_per_class, per_sample_f1
from src.common.tiling import TileConfig


# ---------------------------------------------------------------------------
# Corruption battery.
#
# Each operates on the (B, 3, H, W) uint8 frame batch that
# extract_features_multicrop_gpu hands to `frame_transform`, i.e. on the raw
# frame BEFORE illumination correction. That is both the physically correct
# place -- an acquisition artefact is present in what the sensor produces, and
# the correction then has to cope with it -- and the paper's own feature path,
# so absolute numbers stay comparable to the published ones.
#
# uint8 in, uint8 out: a real detector quantises, and gpu_normalize_illumination
# expects to do the .float() itself.
# ---------------------------------------------------------------------------

def _u8t(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0, 255).to(torch.uint8)


def illum_gradient(x: torch.Tensor, s: float) -> torch.Tensor:
    """Multiplicative linear ramp: condenser / Kohler misalignment."""
    W = x.shape[-1]
    ramp = torch.linspace(1.0 - s, 1.0 + s, W, device=x.device, dtype=torch.float32)
    return _u8t(x.float() * ramp.view(1, 1, 1, W))


def vignette(x: torch.Tensor, s: float) -> torch.Tensor:
    """Radial falloff: different objective / condenser aperture."""
    H, W = x.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=torch.float32).view(1, 1, 1, W)
    r2 = yy ** 2 + xx ** 2
    return _u8t(x.float() * (1.0 - s * (r2 / 2.0)))


def brightness(x: torch.Tensor, s: float) -> torch.Tensor:
    """Global intensity scale: lamp voltage / exposure time."""
    return _u8t(x.float() * s)


def contrast(x: torch.Tensor, s: float) -> torch.Tensor:
    """Gamma: detector response curve."""
    return _u8t(((x.float() / 255.0).clamp_(0, 1) ** s) * 255.0)


def defocus(x: torch.Tensor, s: float) -> torch.Tensor:
    """Separable Gaussian blur: focus drift, a real microscopy failure mode."""
    C = x.shape[1]
    radius = max(1, int(round(3.0 * s)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * s ** 2))
    g = g / g.sum()
    k = 2 * radius + 1
    kh = g.view(1, 1, 1, k).expand(C, 1, 1, k).contiguous()
    kv = g.view(1, 1, k, 1).expand(C, 1, k, 1).contiguous()
    f = x.float()
    f = F.conv2d(f, kh, padding=(0, radius), groups=C)
    f = F.conv2d(f, kv, padding=(radius, 0), groups=C)
    return _u8t(f)


def sensor_noise(x: torch.Tensor, s: float) -> torch.Tensor:
    """Additive Gaussian: different camera or gain setting.

    Generator is seeded per condition by run_condition so a condition is
    reproducible and the two illumination arms see identical noise.
    """
    noise = torch.randn(x.shape, device=x.device, dtype=torch.float32,
                        generator=_NOISE_GEN[0]) * s
    return _u8t(x.float() + noise)


def jpeg(x: torch.Tensor, s: float) -> torch.Tensor:
    """Re-encode: different capture software / archive settings.

    The only member that has to leave the GPU -- JPEG is a CPU codec.
    """
    arr = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, format="JPEG", quality=int(s))
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"))
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().to(x.device)


# Mutable single-slot holder so sensor_noise can be reseeded per condition.
_NOISE_GEN: List[Optional[torch.Generator]] = [None]

# name -> (fn, [severity params]) ordered light -> severe
CORRUPTIONS: Dict[str, Tuple[Callable, List[float]]] = {
    "illum_gradient": (illum_gradient, [0.15, 0.30, 0.50]),
    "vignette":       (vignette,       [0.20, 0.40, 0.60]),
    "brightness":     (brightness,     [0.85, 0.70, 0.55]),
    "contrast":       (contrast,       [1.30, 1.60, 2.00]),
    "defocus":        (defocus,        [1.00, 2.00, 3.50]),
    "sensor_noise":   (sensor_noise,   [5.00, 12.0, 25.0]),
    "jpeg":           (jpeg,           [60, 40, 25]),
}

# The two corruptions the divide-by-Gaussian step is specifically supposed to
# remove. Used only for reporting.
ILLUM_FAMILY = {"illum_gradient", "vignette", "brightness"}


def make_frame_transform(corr_name: str, severity: int, device) -> Optional[Callable]:
    """Build the frame_transform hook for one condition, reseeding the noise RNG."""
    if corr_name == "clean":
        return None
    fn, sev_list = CORRUPTIONS[corr_name]
    s = sev_list[severity - 1]
    gen = torch.Generator(device=device)
    gen.manual_seed(1337 + 97 * severity + sum(map(ord, corr_name)))
    _NOISE_GEN[0] = gen
    return lambda x, _fn=fn, _s=s: _fn(x, _s)


def make_resize_transform(tile_size: int) -> Callable:
    """Whole frame downscaled to one tile: the 'why not just resize?' control."""
    def _fn(x: torch.Tensor) -> torch.Tensor:
        f = F.interpolate(x.float(), size=(tile_size, tile_size),
                          mode="bilinear", align_corners=False)
        return _u8t(f)
    return _fn


# ---------------------------------------------------------------------------
# Decoders. All three are frozen; we only score.
# ---------------------------------------------------------------------------

def load_decoders(a_dir: Path, b_dir: Path, c_dir: Path, device) -> dict:
    from src.simplex_unmixing.model import ModelConfig, UnmixerModel
    from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig
    from src.mc_channel.model import MCChannelHead, MCConfig

    out = {}

    cfg_a = load_json(str(a_dir / "config.json"))
    m_a = UnmixerModel(ModelConfig(embedding_dim=cfg_a["embedding_dim"],
                                   num_prototypes=cfg_a["num_prototypes"],
                                   temperature=cfg_a["temperature"]))
    m_a.load_state_dict(torch.load(a_dir / "phoebi_model.pt", map_location=device, weights_only=True))
    out["A"] = (m_a.to(device).eval(), np.asarray(cfg_a["thresholds"], dtype=np.float64), cfg_a)

    cfg_b = load_json(str(b_dir / "config.json"))
    m_b = PrototypeMatchingModel(ProtoConfig(embedding_dim=cfg_b["embedding_dim"],
                                             num_prototypes=cfg_b["num_prototypes"],
                                             thresholds=cfg_b.get("thresholds"),
                                             unknown_threshold=cfg_b.get("unknown_threshold", 0.5)))
    m_b.load_state_dict(torch.load(b_dir / "proto_model.pt", map_location=device, weights_only=True))
    out["B"] = (m_b.to(device).eval(), np.asarray(cfg_b["thresholds"], dtype=np.float64), cfg_b)

    cfg_c = load_json(str(c_dir / "config.json"))
    m_c = MCChannelHead(MCConfig(**cfg_c["mc_config"]))
    # Method C checkpoints wrap the weights: {"state_dict": ..., "config": ...}
    m_c.load_state_dict(torch.load(c_dir / "mc_model.pt", map_location=device)["state_dict"])
    out["C"] = (m_c.to(device).eval(),
                np.asarray(cfg_c["presence_thresholds"], dtype=np.float64), cfg_c)
    return out


def score_decoder(name: str, model, thresholds, features, image_index, n_images, K, device):
    """features are already L2-normalised (N*T, D). Returns (scores, y_pred)."""
    with torch.no_grad():
        feats = features.to(device)
        if name == "A":
            _, weights_tile, _ = model(feats)
            img = scatter_mean_by_image(weights_tile.cpu(), image_index, n_images)
        elif name == "B":
            sims_tile, _ = model(feats)
            img = scatter_mean_by_image(sims_tile.cpu(), image_index, n_images)
        else:
            logits_tile = model(feats)
            img = scatter_mean_by_image(torch.sigmoid(logits_tile).cpu(), image_index, n_images)
    scores = img[:, :K].numpy().astype(np.float64)
    return scores, (scores > thresholds[:K]).astype(np.int64)


# ---------------------------------------------------------------------------

def stratified_subsample(paths, labels, combos, n_target, seed=1337):
    """Even draw per combination, so every combo and every order is represented."""
    by_combo = defaultdict(list)
    for i, c in enumerate(combos):
        by_combo[c].append(i)
    rng = np.random.default_rng(seed)
    per = max(1, n_target // max(1, len(by_combo)))
    keep: List[int] = []
    for c in sorted(by_combo):
        idx = np.asarray(by_combo[c])
        take = min(per, len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep = sorted(keep)
    return ([paths[i] for i in keep], labels[keep],
            [combos[i] for i in keep], np.asarray(keep))


def label_noise_bound(y_true, y_pred, combos, class_names) -> dict:
    """FNR on pure cultures (model error only) vs mixtures (model error + label noise)."""
    combos = np.asarray(combos)
    order = np.asarray([len(c.split("_")) for c in combos])
    out = {}
    for k, name in enumerate(class_names):
        pos = y_true[:, k] == 1
        single = pos & (order == 1)
        mixed = pos & (order > 1)
        def _fnr(mask):
            if mask.sum() == 0:
                return None, 0
            return float((y_pred[mask, k] == 0).mean()), int(mask.sum())
        fnr_s, n_s = _fnr(single)
        fnr_m, n_m = _fnr(mixed)
        out[name] = {
            "fnr_singleton": fnr_s, "n_singleton": n_s,
            "fnr_mixture": fnr_m, "n_mixture": n_m,
            "label_noise_upper_bound": (None if fnr_s is None or fnr_m is None
                                        else float(max(0.0, fnr_m - fnr_s))),
        }
    vals = [v["label_noise_upper_bound"] for v in out.values()
            if v["label_noise_upper_bound"] is not None]
    out["_aggregate"] = {
        "mean_upper_bound": float(np.mean(vals)) if vals else None,
        "max_upper_bound": float(np.max(vals)) if vals else None,
        "interpretation": (
            "Per-field rate at which a culture-derived label may overstate presence. "
            "UPPER bound: mixtures are also intrinsically harder than pure cultures, "
            "and that difficulty is included in the gap."
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--simplex_dir", default="outputs/simplex_unmixing/default")
    ap.add_argument("--proto_dir", default="outputs/prototype_matching/default")
    ap.add_argument("--mc_dir", default="outputs/mc_channel/default")
    ap.add_argument("--output_dir", default="outputs/acquisition_robustness")
    ap.add_argument("--n_sweep", type=int, default=4000,
                    help="stratified subsample size for the corruption sweep")
    ap.add_argument("--full_clean", action="store_true",
                    help="also run clean on the FULL split (headline reproduction "
                         "+ the label-noise bound at full sample size)")
    ap.add_argument("--illuminations", default="divide,none")
    ap.add_argument("--corruptions", default="",
                    help="comma-separated subset of the battery (default: all). "
                         "Use for smoke runs, e.g. --corruptions illum_gradient")
    ap.add_argument("--severities", default="1,2,3")
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_resize_probe", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  output={out_dir}", flush=True)

    paths_all, labels_all, class_names, combos_all = load_real_split(args.splits_path, args.split)
    K = labels_all.shape[1]
    print(f"split={args.split}: {len(paths_all)} images, K={K}, combos={len(set(combos_all))}",
          flush=True)

    dec = load_decoders(Path(args.simplex_dir), Path(args.proto_dir), Path(args.mc_dir), device)
    base_cfg = dec["A"][2]
    tile_cfg = TileConfig(**base_cfg["tile_config"])
    backbone = base_cfg["backbone"]
    illum_sigma = base_cfg.get("illum_sigma", 64.0)
    print(f"backbone={backbone} tile={tile_cfg.tile_size} grid={tile_cfg.eval_grid_size}", flush=True)

    p_s, l_s, c_s, _ = stratified_subsample(paths_all, labels_all, combos_all, args.n_sweep)
    print(f"sweep subsample: {len(p_s)} images "
          f"({len(p_s)//max(1,len(set(c_s)))} per combo)", flush=True)

    rows: List[dict] = []
    noise_report = None

    def run_condition(tag, corr_name, severity, illum, paths, labels, combos, tcfg,
                      frame_transform="auto"):
        """One (corruption, severity, illumination) cell.

        Always goes through extract_features_multicrop_gpu -- the same path that
        produced the published numbers -- with the corruption injected as
        frame_transform, i.e. on the raw uint8 frame before illumination.
        """
        nonlocal noise_report
        ft = (make_frame_transform(corr_name, severity, device)
              if frame_transform == "auto" else frame_transform)

        t0 = time.time()
        feats, image_index = extract_features_multicrop_gpu(
            image_paths=paths, tile_config=tcfg, backbone=backbone,
            frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
            device=device, illum_sigma=illum_sigma, illum_method=illum,
            cache_path=None, frame_transform=ft,
        )
        feats = F.normalize(feats, p=2, dim=1)
        n_img = len(paths)

        for dname in ("A", "B", "C"):
            model, thr, _ = dec[dname]
            scores, y_pred = score_decoder(dname, model, thr, feats, image_index,
                                           n_img, K, device)
            pc = macro_f1_per_class(labels, y_pred, class_names)
            rows.append({
                "condition": tag, "corruption": corr_name, "severity": severity,
                "illumination": illum, "decoder": dname, "n_images": n_img,
                "per_sample_f1": round(float(per_sample_f1(labels, y_pred)), 4),
                "macro_f1": round(float(pc["macro"]), 4),
                "exact_match": round(float(exact_match_accuracy(labels, y_pred)), 4),
                "illum_family": corr_name in ILLUM_FAMILY,
            })
            if tag == "clean_full/divide" and dname == "B":
                noise_report = label_noise_bound(labels, y_pred, combos, class_names)
        print(f"  [{time.time()-t0:6.1f}s] {tag:38s} "
              + "  ".join(f"{r['decoder']}={r['per_sample_f1']:.4f}" for r in rows[-3:]),
              flush=True)

    # 1. clean on the full split -- reproduces the headline and gives the
    #    label-noise bound at full sample size.
    if args.full_clean:
        for illum in args.illuminations.split(","):
            run_condition(f"clean_full/{illum}", "clean", 0, illum,
                          paths_all, labels_all, combos_all, tile_cfg)

    # 2. corruption sweep on the stratified subsample.
    selected = ([c for c in args.corruptions.split(",") if c]
                if args.corruptions else list(CORRUPTIONS))
    unknown = [c for c in selected if c not in CORRUPTIONS]
    if unknown:
        raise SystemExit(f"unknown corruption(s) {unknown}; available: {list(CORRUPTIONS)}")
    sevs = [int(s) for s in args.severities.split(",") if s]
    for illum in args.illuminations.split(","):
        run_condition(f"clean/{illum}", "clean", 0, illum, p_s, l_s, c_s, tile_cfg)
        for corr in selected:
            for sev in sevs:
                run_condition(f"{corr}@{sev}/{illum}", corr, sev, illum,
                              p_s, l_s, c_s, tile_cfg)

    # 3. tiling vs resize: whole frame downscaled to one tile instead of a
    #    4x4 grid of native-resolution tiles. Same decoders, same illumination.
    if not args.skip_resize_probe:
        resize_cfg = TileConfig(**{**base_cfg["tile_config"], "eval_grid_size": 1})
        run_condition("resize_1tile/divide", "resize_1tile", 0, "divide",
                      p_s, l_s, c_s, resize_cfg,
                      frame_transform=make_resize_transform(tile_cfg.tile_size))

    save_json(str(out_dir / "results.json"), rows)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if noise_report is not None:
        save_json(str(out_dir / "label_noise.json"), noise_report)

    # ---- summary -----------------------------------------------------------
    def get(tag, d):
        for r in rows:
            if r["condition"] == tag and r["decoder"] == d:
                return r["per_sample_f1"]
        return None

    lines = ["# Acquisition robustness / illumination attribution / label noise", ""]
    lines.append(f"Split `{args.split}`; sweep n={len(p_s)} stratified over "
                 f"{len(set(c_s))} combinations; decoders frozen.")
    lines.append("")
    lines.append("## Absolute anchors")
    lines.append("")
    lines.append("| condition | n | A | B | C |")
    lines.append("|---|---|---|---|---|")
    for tag in ("clean_full/divide", "clean_full/none", "clean/divide", "clean/none"):
        vals = [get(tag, d) for d in ("A", "B", "C")]
        if all(v is None for v in vals):
            continue
        n = next((r["n_images"] for r in rows if r["condition"] == tag), "--")
        lines.append(f"| {tag} | {n} | "
                     + " | ".join("--" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines.append("")
    lines.append("Published random-split test F1: A 0.6086, B 0.6095, C 0.6740. Every "
                 "condition here runs through extract_features_multicrop_gpu -- the same "
                 "feature path that produced those numbers -- with the corruption injected "
                 "as frame_transform on the raw uint8 frame, before illumination "
                 "correction. `clean_full/divide` should therefore reproduce the published "
                 "row, and `clean/divide` shows whether the stratified subsample is "
                 "unbiased.")
    lines.append("")
    lines.append("## Mean degradation from clean (per-sample F1), by illumination arm")
    lines.append("")
    lines.append("| corruption | family | A divide | A none | B divide | B none | C divide | C none |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for corr in list(CORRUPTIONS) + (["resize_1tile"] if not args.skip_resize_probe else []):
        cells = []
        for d in ("A", "B", "C"):
            for il in ("divide", "none"):
                base = get(f"clean/{il}", d)
                if corr == "resize_1tile":
                    v = get("resize_1tile/divide", d) if il == "divide" else None
                    cells.append("--" if v is None or base is None else f"{v-base:+.4f}")
                    continue
                vs = [get(f"{corr}@{s}/{il}", d) for s in (1, 2, 3)]
                vs = [v for v in vs if v is not None]
                cells.append("--" if not vs or base is None
                             else f"{float(np.mean(vs))-base:+.4f}")
        fam = "illum" if corr in ILLUM_FAMILY else ""
        lines.append(f"| {corr} | {fam} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Negative = worse than clean. The `illum` rows are what the "
                 "divide-by-Gaussian step exists to remove.")
    lines.append("")
    lines.append("**Read the two arms differently.** The `divide` columns are the "
                 "deployed pipeline: trained and tested with the correction, corruption "
                 "applied to the raw frame in between. Those are the robustness numbers. "
                 "The `none` columns strip the correction *at test time only*, against "
                 "prototypes and thresholds that were calibrated with it -- a "
                 "train/test mismatch, so they measure how load-bearing the correction "
                 "is, NOT what a none-trained pipeline would score. For the matched "
                 "train-and-test-without-correction comparison see "
                 "outputs/ablations/illumination (in-distribution) and the separately "
                 "trained none-illumination models.")
    if noise_report:
        agg = noise_report["_aggregate"]
        lines += ["", "## Label-noise upper bound (Method B, full clean split)", "",
                  "| species | FNR pure culture | FNR mixture | upper bound |",
                  "|---|---|---|---|"]
        for name in class_names:
            v = noise_report[name]
            fs = "--" if v["fnr_singleton"] is None else f"{v['fnr_singleton']:.4f} (n={v['n_singleton']})"
            fm = "--" if v["fnr_mixture"] is None else f"{v['fnr_mixture']:.4f} (n={v['n_mixture']})"
            ub = "--" if v["label_noise_upper_bound"] is None else f"{v['label_noise_upper_bound']:.4f}"
            lines.append(f"| {name} | {fs} | {fm} | {ub} |")
        lines += ["", f"mean upper bound {agg['mean_upper_bound']}, "
                      f"max {agg['max_upper_bound']}", "", agg["interpretation"]]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/results.json, results.csv, summary.md", flush=True)


if __name__ == "__main__":
    main()
