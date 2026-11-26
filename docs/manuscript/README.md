# DMBD technical manuscript

The ordered Markdown, configuration, preamble, and references in this
directory form the canonical source. The study uses the checked-in
synthetic detector fixtures. The [earlier audit](../archive/manuscript_scope_audit.md)
and [historical outputs](../../DMBD/output/) remain available.

## Build from a clean checkout

Install Git, uv, Pandoc `3.14`, and XeLaTeX with the
[Template PDF prerequisites](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/guides/startup-and-setup.md).
Run from the standalone repository root with the desired commit checked
out. The disposable working checkout keeps manuscript outputs separate
from the retained data under `DMBD/output/`.

```bash
dmbd_source="$(pwd -P)"
dmbd_commit="$(git rev-parse HEAD)"
render_workspace="$(mktemp -d)"
template_checkout="$render_workspace/template"
git clone https://github.com/docxology/template.git "$template_checkout"
git -C "$template_checkout" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$template_checkout/projects/working"
git clone "$dmbd_source" "$template_checkout/projects/working/Markov_Blanket_Detection"
cd "$template_checkout/projects/working/Markov_Blanket_Detection"
git checkout --detach "$dmbd_commit"
printf '\n/output/\n' >> .git/info/exclude
uv venv --python 3.12 DMBD/.venv
uv pip install --python DMBD/.venv/bin/python \
  --index https://download.pytorch.org/whl/cpu -e ./DMBD
```

Run the observed-versus-expected fixture comparison. The JSON preserves
both role lists. The matrix figure compares their unions.

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/DMBD" \
  DMBD/.venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np

fixture = runpy.run_path("DMBD/tests/test_markov_blanket.py")
static = fixture["TestMarkovBlanket"]()
static.setUp()
dynamic = fixture["TestDynamicMarkovBlanket"]()
dynamic.setUp()

roles = ("parents", "children", "spouses")
expected_roles = (static.expected_parents, static.expected_children, static.expected_spouses)
names = list(static.data.columns)
rows = []
for i, name in enumerate(names):
    expected = {role: sorted(values[i]) for role, values in zip(roles, expected_roles)}
    observed = static.mb.classify_nodes(i)
    expected_members = sorted({node for members in expected.values() for node in members})
    rows.append({
        "target": name,
        "expected_members": expected_members,
        "estimated_members": sorted(observed["blanket"]),
        "expected_roles": expected,
        "estimated_roles": {role: sorted(observed[role]) for role in roles},
        "lag_members": dynamic.dmb.detect_dynamic_blanket(i)["dynamic_classifications"]["lag_1"]["blanket"],
    })

membership = sum(row["expected_members"] == row["estimated_members"] for row in rows)
orientation = sum(row["expected_roles"] == row["estimated_roles"] for row in rows)
lag_count = sum(bool(row["lag_members"]) for row in rows)
lag_shape = list(dynamic.dmb.lagged_data.shape)
assert (membership, orientation, lag_count, lag_shape) == (5, 2, 0, [80, 10])

data = Path("output/data")
figures = Path("output/figures")
data.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)
receipt = {
    "fixture_seed": 42,
    "static_rows": len(static.data),
    "targets": len(names),
    "membership_matches": membership,
    "role_matches": orientation,
    "targets_with_lag_members": lag_count,
    "lag_shape": lag_shape,
    "targets_detail": rows,
}
(data / "fixture_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4), layout="constrained")
for axis, (key, title) in zip(axes, (("expected_members", "Expected"), ("estimated_members", "Estimated"))):
    matrix = np.array([[int(i in row[key]) for i in range(len(names))] for row in rows])
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=range(len(names)), xticklabels=names,
             yticks=range(len(names)), yticklabels=names,
             xlabel="Candidate member", ylabel="Target", title=title)
    for r, c in zip(*np.nonzero(matrix)):
        axis.text(c, r, "●", color="white", ha="center", va="center")
fig.suptitle("Blanket membership in the seeded DMBD fixture")
fig.savefig(figures / "fixture_membership.png", dpi=180)
PYTHON

cd "$template_checkout"
uv sync --locked
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/Markov_Blanket_Detection
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/Markov_Blanket_Detection
```

The working project contains the [fixture results](03_results.md) as
`output/data/fixture_receipt.json` and renders the PDF and HTML under
`output/pdf/` and `output/web/`. Validation and source binding reports
are under `output/reports/`. The broader
[native test command](../USAGE.md#verify-locally) reports the existing
role and cognitive-test discrepancies separately.
