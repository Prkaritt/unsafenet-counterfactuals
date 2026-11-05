"""Helpers for the UnsafeNet video counterfactual feasibility experiment.

Scope: deterministic preprocessing, run manifests, and comparison utilities for a
prompt-guided CogVideoX video-to-video editing baseline. This package does not
implement LD-ViCE and does not verify that any generated clip is safe.

Modules import torch, PIL and numpy lazily so that the preprocessing and manifest
logic can be unit-tested on a machine with no GPU stack installed.
"""

__all__ = ["preprocess", "manifest", "compare"]
__version__ = "0.1.0"
