# Setup and input data

From the repository root, use an isolated Python environment. The commands
below use Python 3.12 and the official CPU PyTorch package index. Package
dependencies come from [DMBD/pyproject.toml](../DMBD/pyproject.toml).

```bash
uv venv --python 3.12 DMBD/.venv
uv pip install --python DMBD/.venv/bin/python \
  --index https://download.pytorch.org/whl/cpu -e ./DMBD
source DMBD/.venv/bin/activate
```

## Inspect a static blanket

Pass a DataFrame with one numeric variable per column and one observation
per row. The target index refers to the column's zero-based position.
Use finite numerical values and enough nonconstant observations for
correlation tests. CPU tensors can also be passed directly.

```python
import numpy as np
import pandas as pd
from framework.markov_blanket import MarkovBlanket

rng = np.random.default_rng(7)
x = rng.normal(size=128)
data = pd.DataFrame({
    "input": x,
    "response": x + 0.2 * rng.normal(size=128),
    "control": rng.normal(size=128),
})
detector = MarkovBlanket(data, alpha=0.05)
print(detector.classify_nodes(0))
```

The returned `blanket` contains the union of the reported parent, child,
and spouse lists. It is the useful starting point for inspecting
statistical membership. The [fixture results](manuscript/03_results.md)
show where the role labels differ from the known graph. Independent
observation, correlation, and conditional fit are distinct from a causal
intervention.

## Inspect time-indexed inputs

Pass `time_column` and `lag` to
[`DynamicMarkovBlanket`](../DMBD/framework/markov_blanket.py).
The constructor sorts distinct timestamps, pairs observations by their
row order within each time group, and builds current and lagged columns.
Use equal group sizes and a consistent row order. It does not join
longitudinal records by identity. Without a time column, rows are treated
as successive times.

The current `lag_1` result does not establish an absence of temporal
dependence. The implementation dispatches lagged indices to a test
bounded by the original number of variables, so these indices are
treated as out of range. Inspect the assembled `lagged_data` separately.

## Verify locally

Use ffmpeg for tests that save MP4 files and a headless Matplotlib backend.

```bash
cd DMBD
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

The audited base has existing failures in the parent/child role
assertions and in two cognitive-identification assertions. The native
synthetic probe in the [manuscript guide](manuscript/README.md) reports
blanket membership and role agreement separately. Avoid writing new
analysis over the tracked `DMBD/output/` archive.
