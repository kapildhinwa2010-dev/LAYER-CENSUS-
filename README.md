# LAYER-CENSUS

Point it at any list of weight matrices and get a full internal atlas of the model.

LAYER-CENSUS produces one atlas entry per layer: shape, parameter count, Frobenius norm, effective rank, and spectrum decay measured as top-1 and top-10 variance shares. The full census adds totals, the layer with the lowest rank, and the largest layer.

## What it does

- Per-layer shape, parameter count, and norm statistics.
- Effective rank via the participation ratio of squared singular values.
- Spectrum decay via top-1 and top-10 variance shares.
- Full census with totals, lowest-rank layer, and largest layer.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from layer_census.census import census

weights = [np.load(f) for f in ["w0.npy", "w1.npy", "w2.npy"]]
atlas = census(weights, names=["q_proj", "k_proj", "v_proj"])
for entry in atlas["layers"]:
    print(entry["name"], entry["shape"], entry["effective_rank"])
```

## How it works

Every layer is decomposed by SVD. The effective rank is the participation ratio of the squared singular values: 1 for a rank-one matrix, equal to the matrix width for a fully balanced spectrum. The top-k variance shares show how fast the spectrum decays.

## Honest scope

- The atlas describes weight geometry; it says nothing about activations or behavior.
- Layers can have different shapes; only statistics, not direct comparisons, are cross-layer.
- Tests use hand-built matrices, so no model download is needed.

## License

MIT