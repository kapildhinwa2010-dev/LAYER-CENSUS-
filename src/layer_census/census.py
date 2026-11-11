"""Build a full internal atlas of model weight matrices.

Point LAYER-CENSUS at a list of 2-D weight matrices and get one
atlas entry per layer: shape, parameter count, Frobenius norm,
effective rank, and spectrum decay (top-1 and top-10 variance
shares). Pure NumPy.
"""

import numpy as np


def effective_rank(matrix):
    """Participation ratio of the squared singular values."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D")
    s = np.linalg.svd(m, compute_uv=False)
    if float(np.sum(s)) < 1e-12:
        return 0.0
    lam = s ** 2
    return float(np.sum(lam) ** 2 / np.sum(lam ** 2))


def layer_stats(weight, name=None):
    """One atlas entry for a single weight matrix."""
    w = np.asarray(weight, dtype=float)
    if w.ndim != 2:
        raise ValueError("weight must be a 2-D matrix")
    s = np.linalg.svd(w, compute_uv=False)
    lam = np.clip(s, 1e-12, None) ** 2
    total = float(np.sum(lam))
    top1 = float(lam[0]) / total if total > 0 else 0.0
    k = min(10, len(lam))
    topk = float(np.sum(lam[:k])) / total if total > 0 else 0.0
    return {
        "name": name,
        "shape": [int(w.shape[0]), int(w.shape[1])],
        "params": int(w.size),
        "frobenius_norm": float(np.linalg.norm(w)),
        "effective_rank": effective_rank(w),
        "top1_variance_share": top1,
        "top10_variance_share": topk,
        "max_abs_weight": float(np.max(np.abs(w))),
    }


def census(weights, names=None):
    """Full atlas for a list of weight matrices."""
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise ValueError("weights must be a non-empty list")
    if names is not None and len(names) != len(weights):
        raise ValueError("names must align with weights")
    entries = [layer_stats(w, name=names[i] if names else None)
               for i, w in enumerate(weights)]
    ranks = [e["effective_rank"] for e in entries]
    return {
        "num_layers": len(entries),
        "total_params": int(sum(e["params"] for e in entries)),
        "mean_effective_rank": float(np.mean(ranks)),
        "lowest_rank_layer": int(np.argmin(ranks)),
        "largest_layer": int(np.argmax([e["params"] for e in entries])),
        "layers": entries,
    }