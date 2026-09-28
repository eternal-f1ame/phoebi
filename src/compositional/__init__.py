"""Compositional-generalisation experiments (September 2026).

Every decoder in the paper reads a whole 224 px tile as one pooled DINOv2 vector (the
CLS token) and explains it as a convex mixture of species prototypes. Held-out per-species
AUROC near chance says that vector is not additive over the cells in the tile. The
experiments in this package change *what* gets unmixed, not the unmixing:

  stage 0  granularity diagnostic: closed-form prototypes at CLS/224, patch-mean/224,
           per-patch/224, CLS/112, CLS/56, with mean / top-quantile / max aggregation
  stage 1  patch-level simplex unmixing (sparsemax per unit, trained prototypes)
  stage 2  learned orthonormal mixing subspace W under the simplex constraint
  stage 3  calibration under shift (thresholds from a nested held-out-combination split)
  mosaic   compositional coverage from real pure-culture sub-crops (training only)

Shared pieces live here; the drivers are experiments/run_comp_*.py; batch wrappers are
slurm/run_comp_*.sh. Nothing in this package touches the paper's outputs.
"""
