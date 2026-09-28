#!/usr/bin/env python3
"""Pillar 3 of PHOEBI: Novel Class Discovery (LOOCV protocol).

For each held-out species k:
  1. Train Method A on K-1 known species (pure-culture init, like the LOOCV
     open-set harness in run_openset_detection.py).
  2. Forward the FULL test split through the K-1 model. Tiles whose residual
     norm exceeds `theta_disc` are flagged as "novel" candidates.
  3. Run greedy cosine clustering on the novel-candidate residuals to PROPOSE
     a set of new prototypes (size variable, controlled by `cluster_similarity`
     and `min_cluster_size`).
  4. Build the full prototype matrix [K-1 known | N_proposed] and re-score the
     test split: each tile -> argmax cosine sim against every prototype.
  5. Per image, the tile-mode prediction is its label (an integer in
     [0, K-1+N_proposed)).

Metrics, per fold:
  - discovery_recall: fraction of held-out test images assigned to ANY of the
    N_proposed new prototypes (i.e., correctly flagged as "not in the K-1
    known set"). This is the closed-form analogue of NCD's clustering-recall.
  - discovery_purity: of the held-out images correctly flagged, what fraction
    land on the SINGLE most-popular new prototype. Hungarian match collapses
    to argmax with 1 held-out class.
  - cluster_accuracy: discovery_recall * discovery_purity. The headline NCD
    number.
  - n_proposed_protos: how many novel prototypes the discovery primitive
    proposed (interpretable signal: ~1 per held-out class is ideal).
  - drift_known_f1_delta: per-class F1 on the K-1 known species, with and
    without the appended new prototypes. Negative = the new prototypes hurt
    known-class performance.

Aggregates mean +/- std across the K folds to
outputs/discovery_loocv/summary.{csv,json}.

Optional: --sweep_thresholds emits a Pareto curve over residual_threshold
(theta_disc) for each fold so we can plot discovery vs drift tradeoff.
"""

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.features import scatter_mean_by_image
from src.common.io import save_json
from src.common.metrics import per_sample_f1, macro_f1_per_class
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.tiling import TileConfig
from src.common.sinkhorn import sinkhorn_cluster
from src.simplex_unmixing.model import (
    ModelConfig,
    UnmixerModel,
    greedy_cosine_clustering,
    initialize_prototypes,
)

from experiments.run_openset_detection import extract_split, subset_by_images


# ---------------------------------------------------------------------------
# Discovery primitive
# ---------------------------------------------------------------------------
def simgcd_propose(
    test_features: torch.Tensor,        # (T_test, D), L2-normalized
    train_features: torch.Tensor,       # (T_train, D), L2-normalized
    train_tile_labels: torch.Tensor,    # (T_train, K_known), {0,1}
    known_prototypes: torch.Tensor,     # (K_known, D), L2-normalized
    K_novel: int,
    residual_threshold: float,
    epochs: int,
    lr: float,
    sk_temperature: float,
    ce_temperature: float,
    w_labeled: float,
    sk_iters: int,
    sk_epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """SimGCD-style adaptation for PHOEBI LOOCV (K_novel = 1 by default).

    Trains a (K_known + K_novel)-way classifier head on top of frozen DINOv2
    features. Loss = w_labeled * BCE(supervised, train tiles)
                  + (1 - w_labeled) * CE(SK soft pseudo-labels, test tiles).

    Initialisation: K_known rows from pure-culture prototypes, K_novel rows
    from SK K=1 centroid on residual-flagged test tiles. The SupCon term in
    the original SimGCD paper is dropped because we operate on frozen
    pre-extracted features (no augmentation pairs).

    Returns the trained novel rows of the head (L2-normalized), shape (K_novel, D).
    """
    K_known = known_prototypes.shape[0]
    D = known_prototypes.shape[1]

    # Init novel rows from SK K=1 centroid on residual-flagged test tiles
    cfg = ModelConfig(embedding_dim=D, num_prototypes=K_known, temperature=10.0)
    teacher = UnmixerModel(cfg).to(device)
    teacher.prototypes.data.copy_(known_prototypes.to(device))
    teacher.eval()
    with torch.no_grad():
        _, _, residuals = teacher(test_features.to(device))
    norms = residuals.norm(p=2, dim=1)
    novel_mask = norms > residual_threshold
    if int(novel_mask.sum()) < K_novel * 5:
        # too few flagged tiles; SimGCD has no signal to start from
        return torch.empty(0, D)

    novel_feats = test_features.to(device)[novel_mask]
    sk_centroids, _ = sinkhorn_cluster(
        novel_feats, num_clusters=K_novel,
        num_iters=30, sk_iters=3, sk_epsilon=0.05, device=device,
    )

    # Full head: known | novel
    head = torch.nn.Parameter(torch.cat([
        known_prototypes.to(device),
        sk_centroids.to(device),
    ], dim=0))  # (K_known + K_novel, D)

    train_feats_dev = train_features.to(device)
    train_tile_labels_dev = train_tile_labels.to(device)  # (T_train, K_known)
    test_feats_dev = test_features.to(device)
    n_train = train_feats_dev.shape[0]
    n_test = test_feats_dev.shape[0]

    optimizer = torch.optim.Adam([head], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    # The supervised BCE applies ONLY to the K_known logits — supervising the
    # novel column with target=0 on every training tile pushes the novel head
    # away from the entire data manifold (off-data direction), so the held-out
    # species (whose tile features lie ON the manifold) never argmax to it.
    # The novel head is shaped solely by the SK pseudo-label CE on test tiles.

    batch_size = 4096
    for ep in range(epochs):
        perm_train = torch.randperm(n_train, device=device)
        running_l = running_u = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm_train[i:i + batch_size]
            h_norm = F.normalize(head, p=2, dim=1)
            # supervised BCE on the K_known known columns ONLY
            x_l = train_feats_dev[idx]
            logits_full = (x_l @ h_norm.t()) / ce_temperature
            logits_l = logits_full[:, :K_known]
            y_l = train_tile_labels_dev[idx]
            loss_l = bce(logits_l, y_l)

            # unsupervised SK pseudo-labels on test tiles
            test_idx = torch.randint(0, n_test, (min(batch_size, n_test),), device=device)
            x_u = test_feats_dev[test_idx]
            with torch.no_grad():
                logits_u_detach = (x_u @ h_norm.t()) / sk_temperature
                # SK over (K_known + K_novel) categories using softmax
                # We balance the assignment via Sinkhorn-Knopp normalisation
                Q = F.softmax(logits_u_detach, dim=1).t()  # (K, B)
                B = Q.shape[1]
                K = Q.shape[0]
                Q = Q / Q.sum()
                for _ in range(sk_iters):
                    Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8) / K
                    Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8) / B
                Q = (Q * B).t()  # (B, K), columns sum to 1
                pseudo = Q.detach()

            logits_u = (x_u @ h_norm.t()) / ce_temperature
            log_p = F.log_softmax(logits_u, dim=1)
            loss_u = -(pseudo * log_p).sum(dim=1).mean()

            loss = w_labeled * loss_l + (1.0 - w_labeled) * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_l += loss_l.item()
            running_u += loss_u.item()

    # Diagnostic: where did the novel head land?
    h_norm_final = F.normalize(head.data, p=2, dim=1)
    novel_norm = h_norm_final[K_known:]
    known_norm = h_norm_final[:K_known]
    # Cosine sim of novel head with each known prototype (diagnostic)
    nv_known_sim = (novel_norm @ known_norm.t()).cpu()
    # Mean sim of novel head with all test tiles
    test_sim_per_head = (test_feats_dev @ h_norm_final.t()).cpu()  # (T_test, K)
    novel_argmax_count = int((test_sim_per_head.argmax(dim=1) >= K_known).sum().item())
    n_test_tiles = test_sim_per_head.shape[0]
    print(f"    [simgcd] novel↔known sim: {nv_known_sim.flatten().tolist()}", flush=True)
    print(f"    [simgcd] {novel_argmax_count}/{n_test_tiles} test TILES argmax to novel "
          f"(novel_max_sim={test_sim_per_head[:, K_known:].max().item():.3f}, "
          f"known_max_sim={test_sim_per_head[:, :K_known].max().item():.3f})", flush=True)

    novel_rows = head.data[K_known:].detach().cpu()
    novel_rows = F.normalize(novel_rows, p=2, dim=1)
    return novel_rows


def propose_new_prototypes(
    test_features: torch.Tensor,
    known_prototypes: torch.Tensor,
    residual_threshold: float,
    cluster_similarity: float,
    min_cluster_size: int,
    temperature: float,
    device: torch.device,
    cluster_method: str = "greedy",
    sinkhorn_k: int = 2,
    train_features: torch.Tensor = None,
    train_tile_labels: torch.Tensor = None,
    simgcd_epochs: int = 30,
    simgcd_lr: float = 1e-2,
    simgcd_w_labeled: float = 0.5,
    simgcd_sk_temperature: float = 0.1,
    simgcd_ce_temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Returns (proposed_prototypes [N, D], all_residuals_per_tile [T, D],
    per_tile_residual_norms [T]).

    Key correctness fix vs the original train.py:--discover path: high residual
    norm flags a tile as novel, but the cluster centroids are computed in
    FEATURE space, not residual space. Prototypes live in feature space
    (cosine-similarity-with-features), so the new ones must too -- otherwise
    the appended cluster directions are unrelated to anything the model can
    later score against."""
    cfg = ModelConfig(
        embedding_dim=known_prototypes.shape[1],
        num_prototypes=known_prototypes.shape[0],
        temperature=temperature,
    )
    model = UnmixerModel(cfg).to(device)
    model.prototypes.data.copy_(known_prototypes.to(device))
    model.eval()

    feats_dev = test_features.to(device)
    with torch.no_grad():
        _, _, residuals = model(feats_dev)
    residuals = residuals.detach().cpu()
    norms = residuals.norm(p=2, dim=1).numpy()
    novel_mask = norms > residual_threshold

    if int(novel_mask.sum()) < min_cluster_size:
        return torch.empty(0, known_prototypes.shape[1]), residuals, norms

    # Cluster the FEATURES of high-residual tiles (not the residuals themselves).
    novel_features = test_features[novel_mask]

    if cluster_method == "sinkhorn":
        # SK k-means with a fixed K -> balanced cluster sizes by construction.
        # Avoids the over-fragmentation failure mode of greedy clustering.
        if novel_features.shape[0] < sinkhorn_k * 5:  # need at least 5 samples per cluster
            return torch.empty(0, known_prototypes.shape[1]), residuals, norms
        centroids, _ = sinkhorn_cluster(
            novel_features.to(device),
            num_clusters=sinkhorn_k,
            num_iters=30,
            sk_iters=3,
            sk_epsilon=0.05,
            device=device,
        )
        return centroids.detach().cpu(), residuals, norms

    if cluster_method == "uno_lite":
        # UNO-spirit baseline (multi-label adaptation): start with SK cluster
        # centroids in feature space, then gradient-refine them with BCE on SK
        # pseudo-labels. The prototypes are the centroids themselves (kept on
        # the unit sphere via repeated L2-normalization), not arbitrary linear
        # weights, so they stay aligned with the known-prototype geometry.
        # This is the multi-label analogue of UNO's classifier-weight prototypes
        # but constrained to the feature manifold the rest of \methodA{} lives
        # in. Not the full UNO scaffold (no swapped prediction, no multi-head)
        # but a meaningfully stronger discovery baseline than vanilla SK because
        # the centroids get pulled toward discriminative directions during BCE.
        K_novel = sinkhorn_k
        if novel_features.shape[0] < K_novel * 5:
            return torch.empty(0, known_prototypes.shape[1]), residuals, norms
        # Step 1: SK pseudo-labels (returns centroids + hard assignments)
        centroids, hard = sinkhorn_cluster(
            novel_features.to(device),
            num_clusters=K_novel,
            num_iters=30,
            sk_iters=3,
            sk_epsilon=0.05,
            device=device,
        )
        # Step 2: Gradient-refine centroids via BCE-on-cosine-sim. Each tile is
        # assigned to one slot (SK hard-label); the gradient pulls the matching
        # centroid toward the tile and pushes the others away.
        proto = torch.nn.Parameter(centroids.clone())  # (K_novel, D), on device
        feats_dev = novel_features.to(device)
        target = torch.nn.functional.one_hot(hard.to(device), num_classes=K_novel).float()
        opt = torch.optim.Adam([proto], lr=1e-2)
        bce = torch.nn.BCEWithLogitsLoss()
        feats_norm = torch.nn.functional.normalize(feats_dev, p=2, dim=1)
        for _ in range(100):
            proto_norm = torch.nn.functional.normalize(proto, p=2, dim=1)
            logits = (feats_norm @ proto_norm.t()) * 10.0  # temperature 10 to scale cosine sim into BCE-meaningful range
            loss = bce(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        proposed = torch.nn.functional.normalize(proto.data, p=2, dim=1)
        return proposed.detach().cpu(), residuals, norms

    if cluster_method == "simgcd":
        if train_features is None or train_tile_labels is None:
            raise ValueError("simgcd requires train_features and train_tile_labels")
        proposed = simgcd_propose(
            test_features=test_features,
            train_features=train_features,
            train_tile_labels=train_tile_labels,
            known_prototypes=known_prototypes,
            K_novel=sinkhorn_k,
            residual_threshold=residual_threshold,
            epochs=simgcd_epochs,
            lr=simgcd_lr,
            sk_temperature=simgcd_sk_temperature,
            ce_temperature=simgcd_ce_temperature,
            w_labeled=simgcd_w_labeled,
            sk_iters=3,
            sk_epsilon=0.05,
            device=device,
        )
        return proposed, residuals, norms

    # default: greedy cosine clustering on features (post-bug-fix)
    proposed = greedy_cosine_clustering(
        novel_features.to(device),
        threshold=cluster_similarity,
        min_size=min_cluster_size,
    )
    if proposed is None or proposed.numel() == 0:
        return torch.empty(0, known_prototypes.shape[1]), residuals, norms
    return proposed.detach().cpu(), residuals, norms


# ---------------------------------------------------------------------------
# Class-relation knowledge distillation (Cr-KD-NCD adaptation)
# ---------------------------------------------------------------------------
def refine_with_class_relation_kd(
    test_features: torch.Tensor,
    known_prototypes: torch.Tensor,
    proposed_prototypes: torch.Tensor,
    kd_lambda: float,
    kd_steps: int,
    kd_lr: float,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Gradient-refine the proposed prototypes with reconstruction loss
    plus a class-relation KD term that forces the full model's simplex
    weights on the known dims to match the frozen K-1 teacher's weights.

    Adapted from Cr-KD-NCD~\\citep{gu2023class}: their KD loss aligns the
    student's labeled-head distribution to a frozen known-only teacher.
    Here we have closed-form simplex weights instead of softmax logits,
    but the structural fix is the same: pin the known-class behavior
    while letting the new dimensions learn.

    Returns the refined proposed prototypes (only the new ones updated;
    the K-1 known prototypes are frozen by zeroing their gradient)."""
    K_known = known_prototypes.shape[0]
    K_new = proposed_prototypes.shape[0]
    if K_new == 0:
        return proposed_prototypes

    # Frozen teacher: K-1 known prototypes
    teacher = UnmixerModel(ModelConfig(
        embedding_dim=known_prototypes.shape[1],
        num_prototypes=K_known,
        temperature=temperature,
    )).to(device)
    teacher.prototypes.data.copy_(known_prototypes.to(device))
    teacher.eval()
    feats_dev = test_features.to(device)
    with torch.no_grad():
        _, teacher_w, _ = teacher(feats_dev)
    teacher_w = teacher_w.detach()

    # Student: full prototype matrix [known | new]; only new are trainable
    student = UnmixerModel(ModelConfig(
        embedding_dim=known_prototypes.shape[1],
        num_prototypes=K_known + K_new,
        temperature=temperature,
    )).to(device)
    full_protos = torch.cat([known_prototypes, proposed_prototypes], dim=0)
    student.prototypes.data.copy_(full_protos.to(device))

    optimizer = torch.optim.Adam([student.prototypes], lr=kd_lr)
    for step in range(kd_steps):
        z_recon, w_full, _ = student(feats_dev)
        loss_rec = F.mse_loss(z_recon, feats_dev)

        # Per-image gate (Cr-KD-NCD): KD strength scaled by how much weight the
        # student already puts on the known dims. For samples the student
        # routes mostly to NEW dims (likely novel), the gate shrinks toward 0
        # so KD does not force them back into the known classes. For samples
        # the student routes to the known dims (likely known), the gate is ~1
        # and KD enforces teacher agreement.
        with torch.no_grad():
            gate = w_full[:, :K_known].sum(dim=1, keepdim=True)  # (T, 1)
            gate = gate / gate.mean().clamp(min=1e-8)
        loss_kd = ((w_full[:, :K_known] - teacher_w) ** 2 * gate).mean()

        loss = loss_rec + kd_lambda * loss_kd
        optimizer.zero_grad()
        loss.backward()
        # Freeze the K_known known prototypes
        student.prototypes.grad[:K_known].zero_()
        optimizer.step()

    return student.prototypes.data[K_known:].detach().cpu()


# ---------------------------------------------------------------------------
# Per-image prototype assignment
# ---------------------------------------------------------------------------
def assign_images_to_prototypes(
    test_features: torch.Tensor,
    test_image_index: torch.Tensor,
    num_images: int,
    prototypes: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each tile: argmax cosine sim against `prototypes`. Aggregate to image
    by averaging per-class similarities, then argmax. Returns
    (image_argmax_proto [N], image_max_sim [N])."""
    z = F.normalize(test_features, p=2, dim=1).to(device)
    p = F.normalize(prototypes, p=2, dim=1).to(device)
    sims = z @ p.t()  # [T, P]
    sims_image = scatter_mean_by_image(sims.cpu(), test_image_index, num_images)
    arg = sims_image.argmax(dim=1).numpy()
    mx = sims_image.max(dim=1).values.numpy()
    return arg, mx


# ---------------------------------------------------------------------------
# Per-image presence prediction (for drift evaluation)
# ---------------------------------------------------------------------------
def presence_prediction_from_unmixer(
    test_features: torch.Tensor,
    test_image_index: torch.Tensor,
    num_images: int,
    prototypes: torch.Tensor,
    K_known: int,
    temperature: float,
    presence_thresholds: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run sparsemax unmixer on `prototypes`, mean-aggregate per image, threshold
    the first K_known prototype weights to get a {0,1}^K_known prediction."""
    cfg = ModelConfig(
        embedding_dim=prototypes.shape[1],
        num_prototypes=prototypes.shape[0],
        temperature=temperature,
    )
    model = UnmixerModel(cfg).to(device)
    model.prototypes.data.copy_(prototypes.to(device))
    model.eval()
    with torch.no_grad():
        _, weights_tile, _ = model(test_features.to(device))
    weights_image = scatter_mean_by_image(weights_tile.cpu(), test_image_index, num_images)
    w_known = weights_image[:, :K_known].numpy()
    return (w_known > presence_thresholds[None, :]).astype(np.int64)


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------
def run_fold(
    held_out: str,
    class_names: List[str],
    train_labels: np.ndarray,
    train_videos: List[str],
    train_features: torch.Tensor,
    train_image_index: torch.Tensor,
    test_labels: np.ndarray,
    test_features: torch.Tensor,
    test_image_index: torch.Tensor,
    args,
    device: torch.device,
) -> Dict:
    k_idx = class_names.index(held_out)
    kept_class_names = [c for i, c in enumerate(class_names) if i != k_idx]
    K_known = len(kept_class_names)

    # ---- 1. filter train, init K-1 known prototypes ----
    train_keep_mask = train_labels[:, k_idx] == 0
    train_keep_image_idx = np.where(train_keep_mask)[0]
    filtered_train_videos = [train_videos[i] for i in train_keep_image_idx]
    filtered_train_features, filtered_train_image_index = subset_by_images(
        train_features, train_image_index, train_keep_image_idx
    )
    known_protos = init_prototypes_from_pure_cultures(
        tile_features=filtered_train_features,
        image_index=filtered_train_image_index,
        video_ids=filtered_train_videos,
        class_names=kept_class_names,
    )
    if known_protos is None:
        print(f"  [fold excl={held_out}] pure-culture init failed; falling back to K-means")
        known_protos = initialize_prototypes(filtered_train_features, K_known, init="kmeans")

    # ---- 2. propose new prototypes from test residuals ----
    # SimGCD path needs per-tile labels on the K-1 known dims.
    train_tile_labels_known = None
    if args.cluster_method == "simgcd":
        # Each tile inherits its parent image's multilabel; column k_idx is dropped.
        train_labels_kept_imgs = train_labels[train_keep_image_idx]
        # The image_index in filtered_train_image_index references the COMPRESSED
        # image space (0..n_filtered-1) because subset_by_images remaps it.
        train_image_labels_known = np.delete(train_labels_kept_imgs, k_idx, axis=1)
        # Each tile -> its image's label vector
        ti_idx = filtered_train_image_index.cpu().numpy()
        train_tile_labels_known = torch.from_numpy(
            train_image_labels_known[ti_idx].astype(np.float32)
        )

    proposed, residuals_all, residual_norms = propose_new_prototypes(
        test_features=test_features,
        known_prototypes=known_protos,
        residual_threshold=args.residual_threshold,
        cluster_similarity=args.cluster_similarity,
        min_cluster_size=args.min_cluster_size,
        temperature=args.temperature,
        device=device,
        cluster_method=args.cluster_method,
        sinkhorn_k=args.sinkhorn_k,
        train_features=filtered_train_features if args.cluster_method == "simgcd" else None,
        train_tile_labels=train_tile_labels_known,
        simgcd_epochs=getattr(args, "simgcd_epochs", 30),
        simgcd_lr=getattr(args, "simgcd_lr", 1e-2),
        simgcd_w_labeled=getattr(args, "simgcd_w_labeled", 0.5),
        simgcd_sk_temperature=getattr(args, "simgcd_sk_temperature", 0.1),
        simgcd_ce_temperature=getattr(args, "simgcd_ce_temperature", 0.1),
    )
    n_proposed = int(proposed.shape[0])

    # ---- 2b. optional Cr-KD-NCD style class-relation KD refinement ----
    if args.kd_lambda > 0 and n_proposed > 0:
        proposed = refine_with_class_relation_kd(
            test_features=test_features,
            known_prototypes=known_protos,
            proposed_prototypes=proposed,
            kd_lambda=args.kd_lambda,
            kd_steps=args.kd_steps,
            kd_lr=args.kd_lr,
            temperature=args.temperature,
            device=device,
        )

    num_test = test_labels.shape[0]
    held_out_mask = (test_labels[:, k_idx] == 1).astype(bool)
    n_held_out_imgs = int(held_out_mask.sum())

    # ---- 3a. discovery metric: full prototype matrix, per-image argmax ----
    if n_proposed == 0:
        discovery_recall = 0.0
        discovery_purity = float("nan")
        cluster_accuracy = 0.0
        n_dominant = 0
    else:
        full_protos = torch.cat([known_protos, proposed], dim=0)
        image_argmax, image_max_sim = assign_images_to_prototypes(
            test_features=test_features,
            test_image_index=test_image_index,
            num_images=num_test,
            prototypes=full_protos,
            device=device,
        )
        held_out_argmax = image_argmax[held_out_mask]
        # Discovery recall: held-out images assigned to any NEW prototype
        # (proto index >= K_known)
        new_assignments = held_out_argmax >= K_known
        discovery_recall = float(new_assignments.mean())

        # Of those, fraction that land on the SINGLE dominant new prototype
        if new_assignments.sum() > 0:
            new_protos_for_held_out = held_out_argmax[new_assignments]
            counts = np.bincount(new_protos_for_held_out, minlength=K_known + n_proposed)
            n_dominant = int(counts[K_known:].max())
            discovery_purity = float(n_dominant / new_assignments.sum())
        else:
            discovery_purity = float("nan")
            n_dominant = 0

        cluster_accuracy = float(n_dominant / n_held_out_imgs) if n_held_out_imgs else 0.0

    # ---- 3b. drift: per-class F1 on K_known known species, before vs after ----
    # Calibrate per-class thresholds on the K-1 known classes (use 5th percentile
    # of weight on positive train images for each known class).
    cfg_known = ModelConfig(
        embedding_dim=known_protos.shape[1],
        num_prototypes=K_known,
        temperature=args.temperature,
    )
    m_known = UnmixerModel(cfg_known).to(device)
    m_known.prototypes.data.copy_(known_protos.to(device))
    m_known.eval()
    with torch.no_grad():
        _, train_w_tile, _ = m_known(filtered_train_features.to(device))
    n_train_imgs = int(filtered_train_image_index.max().item()) + 1 if filtered_train_image_index.numel() else 0
    train_w_image = scatter_mean_by_image(train_w_tile.cpu(), filtered_train_image_index, n_train_imgs).numpy()
    train_labels_kept = train_labels[train_keep_image_idx]
    train_labels_known = np.delete(train_labels_kept, k_idx, axis=1)
    presence_thresholds = np.zeros(K_known)
    for j in range(K_known):
        pos = train_w_image[train_labels_known[:, j] == 1, j]
        presence_thresholds[j] = float(np.quantile(pos, args.calibrate_quantile)) if pos.size else 0.5

    # Test labels collapsed to K-1 known
    test_labels_known = np.delete(test_labels, k_idx, axis=1)

    # F1 BEFORE: only K_known prototypes
    pred_before = presence_prediction_from_unmixer(
        test_features=test_features,
        test_image_index=test_image_index,
        num_images=num_test,
        prototypes=known_protos,
        K_known=K_known,
        temperature=args.temperature,
        presence_thresholds=presence_thresholds,
        device=device,
    )
    f1_before = float(per_sample_f1(test_labels_known, pred_before))
    macro_before = macro_f1_per_class(test_labels_known, pred_before, kept_class_names)

    # F1 AFTER: append proposed prototypes (which can steal weight from known dims)
    if n_proposed > 0:
        full_protos = torch.cat([known_protos, proposed], dim=0)
        pred_after = presence_prediction_from_unmixer(
            test_features=test_features,
            test_image_index=test_image_index,
            num_images=num_test,
            prototypes=full_protos,
            K_known=K_known,
            temperature=args.temperature,
            presence_thresholds=presence_thresholds,
            device=device,
        )
        f1_after = float(per_sample_f1(test_labels_known, pred_after))
        macro_after = macro_f1_per_class(test_labels_known, pred_after, kept_class_names)
    else:
        f1_after = f1_before
        macro_after = dict(macro_before)

    return {
        "held_out": held_out,
        "n_held_out_test_imgs": n_held_out_imgs,
        "n_proposed_protos": n_proposed,
        "discovery_recall": discovery_recall,
        "discovery_purity": discovery_purity,
        "cluster_accuracy": cluster_accuracy,
        "drift_known_f1_before": f1_before,
        "drift_known_f1_after": f1_after,
        "drift_known_f1_delta": f1_after - f1_before,
        "drift_macro_f1_before": macro_before["macro"],
        "drift_macro_f1_after": macro_after["macro"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PHOEBI Pillar 3: NCD harness")
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--output_dir", type=str, default="outputs/discovery_loocv")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")

    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--eval_grid_size", type=int, default=4)
    parser.add_argument("--train_tiles_per_image", type=int, default=16)

    parser.add_argument("--illumination", type=str, default="divide",
                        choices=["divide", "subtract", "none"])
    parser.add_argument("--illum_sigma", type=float, default=64.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--temperature", type=float, default=10.0)

    # discovery primitive hyperparams
    parser.add_argument("--residual_threshold", type=float, default=0.15,
                        help="theta_disc: only tiles with residual_norm > this are clustered")
    parser.add_argument("--cluster_similarity", type=float, default=0.7,
                        help="cosine similarity threshold for greedy cluster expansion")
    parser.add_argument("--min_cluster_size", type=int, default=50,
                        help="min number of high-residual tiles to form a cluster")
    parser.add_argument("--calibrate_quantile", type=float, default=0.05)

    parser.add_argument("--cluster_method", type=str, default="greedy",
                        choices=["greedy", "sinkhorn", "uno_lite", "simgcd"],
                        help="greedy: variable-K cosine clustering. sinkhorn: fixed-K balanced k-means. uno_lite: UNO-spirit (SK pseudo-labels + BCE-trained linear head as prototypes). simgcd: SimGCD-style adapted to PHOEBI LOOCV with K_novel=1 (BCE on labeled train tiles + SK soft pseudo-label CE on test tiles).")
    parser.add_argument("--sinkhorn_k", type=int, default=2,
                        help="number of novel-prototype candidates for SK / SimGCD clustering")
    parser.add_argument("--simgcd_epochs", type=int, default=30,
                        help="SimGCD training epochs over train tiles")
    parser.add_argument("--simgcd_lr", type=float, default=1e-2,
                        help="SimGCD optimizer learning rate")
    parser.add_argument("--simgcd_w_labeled", type=float, default=0.5,
                        help="SimGCD weight on supervised vs SK pseudo-label loss")
    parser.add_argument("--simgcd_sk_temperature", type=float, default=0.1,
                        help="SimGCD softmax temperature for SK pseudo-labels")
    parser.add_argument("--simgcd_ce_temperature", type=float, default=0.1,
                        help="SimGCD softmax temperature for the loss CE term")

    parser.add_argument("--kd_lambda", type=float, default=0.0,
                        help="Cr-KD-NCD class-relation KD weight; 0 disables. Refines new prototypes only.")
    parser.add_argument("--kd_steps", type=int, default=100,
                        help="Number of gradient steps for KD refinement.")
    parser.add_argument("--kd_lr", type=float, default=1e-3,
                        help="Adam lr for KD refinement of new prototypes.")

    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Sweep theta_disc over a Pareto curve per fold")
    parser.add_argument("--sweep_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30",
                        help="comma-separated theta_disc values for the sweep")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.train_tiles_per_image,
        eval_grid_size=args.eval_grid_size,
    )
    print(f"Device: {device} | TileConfig: {tile_cfg}")

    # ---- 1. extract train + test once ----
    print("\n[1/3] Extracting tile features (cached)...")
    (_, train_labels, class_names, train_videos, train_features, train_image_index) = \
        extract_split(
            split="train", splits_path=args.splits_path,
            tile_cfg=tile_cfg, backbone=args.backbone, device=device,
            illum_method=args.illumination, illum_sigma=args.illum_sigma,
            frame_batch_size=args.frame_batch_size, batch_size=args.batch_size,
            num_workers=args.num_workers, cache_dir=features_dir,
        )
    (_, test_labels, _, _, test_features, test_image_index) = extract_split(
        split="test", splits_path=args.splits_path,
        tile_cfg=tile_cfg, backbone=args.backbone, device=device,
        illum_method=args.illumination, illum_sigma=args.illum_sigma,
        frame_batch_size=args.frame_batch_size, batch_size=args.batch_size,
        num_workers=args.num_workers, cache_dir=features_dir,
    )
    print(f"  train tiles: {tuple(train_features.shape)}  test tiles: {tuple(test_features.shape)}")
    print(f"  classes ({len(class_names)}): {class_names}")

    # ---- 2. LOOCV discovery ----
    print(f"\n[2/3] Running discovery LOOCV across {len(class_names)} classes...")
    if args.sweep_thresholds:
        thresholds = [float(x) for x in args.sweep_grid.split(",")]
        sweep_rows = []
        for theta in thresholds:
            args.residual_threshold = theta
            for held_out in class_names:
                fold = run_fold(
                    held_out=held_out, class_names=class_names,
                    train_labels=train_labels, train_videos=train_videos,
                    train_features=train_features, train_image_index=train_image_index,
                    test_labels=test_labels, test_features=test_features,
                    test_image_index=test_image_index,
                    args=args, device=device,
                )
                row = {"theta_disc": theta, **fold}
                sweep_rows.append(row)
                print(f"  theta={theta:.3f} excl={held_out:3s} | "
                      f"recall={fold['discovery_recall']:.3f} "
                      f"purity={fold['discovery_purity']:.3f} "
                      f"acc={fold['cluster_accuracy']:.3f} "
                      f"drift={fold['drift_known_f1_delta']:+.4f} "
                      f"n_proto={fold['n_proposed_protos']}")
        per_fold = sweep_rows
    else:
        per_fold = []
        for held_out in class_names:
            fold = run_fold(
                held_out=held_out, class_names=class_names,
                train_labels=train_labels, train_videos=train_videos,
                train_features=train_features, train_image_index=train_image_index,
                test_labels=test_labels, test_features=test_features,
                test_image_index=test_image_index,
                args=args, device=device,
            )
            per_fold.append(fold)
            print(f"  excl={held_out:3s} | "
                  f"n_held_out={fold['n_held_out_test_imgs']:4d} "
                  f"n_proto={fold['n_proposed_protos']:2d} "
                  f"recall={fold['discovery_recall']:.3f} "
                  f"purity={fold['discovery_purity']:.3f} "
                  f"acc={fold['cluster_accuracy']:.3f} "
                  f"drift={fold['drift_known_f1_delta']:+.4f}")

    # ---- 3. aggregate ----
    print("\n[3/3] Aggregating...")
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        writer.writeheader()
        writer.writerows(per_fold)

    def mean_std(key):
        vals = np.array([
            r[key] for r in per_fold
            if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))
        ])
        return (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), float("nan"))

    summary = {
        "class_names": class_names,
        "num_folds": len(per_fold),
        "discovery_recall": dict(zip(("mean", "std"), mean_std("discovery_recall"))),
        "discovery_purity": dict(zip(("mean", "std"), mean_std("discovery_purity"))),
        "cluster_accuracy": dict(zip(("mean", "std"), mean_std("cluster_accuracy"))),
        "drift_known_f1_delta": dict(zip(("mean", "std"), mean_std("drift_known_f1_delta"))),
        "n_proposed_protos": dict(zip(("mean", "std"), mean_std("n_proposed_protos"))),
        "per_fold": per_fold,
        "config": {
            "splits_path": args.splits_path, "backbone": args.backbone,
            "tile_config": asdict(tile_cfg),
            "illumination": args.illumination, "illum_sigma": args.illum_sigma,
            "temperature": args.temperature,
            "residual_threshold": args.residual_threshold,
            "cluster_similarity": args.cluster_similarity,
            "min_cluster_size": args.min_cluster_size,
            "sweep_thresholds": args.sweep_thresholds,
        },
    }
    save_json(str(output_dir / "summary.json"), summary)

    print(f"\n=== Discovery LOOCV summary ({len(per_fold)} rows) ===")
    print(f"  cluster_accuracy: {summary['cluster_accuracy']['mean']:.4f} +/- {summary['cluster_accuracy']['std']:.4f}")
    print(f"  discovery_recall: {summary['discovery_recall']['mean']:.4f} +/- {summary['discovery_recall']['std']:.4f}")
    print(f"  discovery_purity: {summary['discovery_purity']['mean']:.4f} +/- {summary['discovery_purity']['std']:.4f}")
    print(f"  drift_F1_delta:   {summary['drift_known_f1_delta']['mean']:+.4f} +/- {summary['drift_known_f1_delta']['std']:.4f}")
    print(f"  n_proposed:       {summary['n_proposed_protos']['mean']:.2f} +/- {summary['n_proposed_protos']['std']:.2f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
