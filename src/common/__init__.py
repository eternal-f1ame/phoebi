"""Shared utilities for PHOEBI (Bacterial Identification via Sparse Prototype Decomposition).

Both Method A (simplex unmixing) and Method B (prototype matching) are built on the
spatial homogeneity assumption: for any sufficiently large crop c of a microscopy frame
x, P(y|c) = P(y|x). This makes random crops label-preserving augmentations and lets us
aggregate per-tile predictions by simple averaging. See `tiling.py` for the
implementation and the per-method `model.py` files for the math each method runs on top.
"""
