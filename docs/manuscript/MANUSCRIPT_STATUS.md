# Manuscript status — Markov_Blanket_Detection (DMBD)

**Repo type:** Tool/library. A PyTorch framework for dynamic Markov blanket
detection (`DMBD/pyproject.toml`: name `dmbd`, "Dynamic Markov Blanket
Detection Framework"), with a gridworld demo module and generated analysis
outputs under `DMBD/output/`.

**Evidence checked:** repo root listing (`README.md`, `LICENSE.md`, `DMBD/`),
`DMBD/pyproject.toml`, `DMBD/README_gridworld.md`, `DMBD/framework/` module
list, `DMBD/output/`. No `manuscript/` or `docs/manuscript/` directory
existed before this file; the repo ships no paper source, only code,
tests, and generated reports/visualizations.

**Why no publication-target manuscript applies today:** the repository is a
detection framework implementation, not a narrative research output; its
deliverables are code, tests, and demo analyses.

**What would trigger creating one:** a methods paper on the DMBD framework
itself (e.g. extending Beck & Ramstead 2025, arXiv:2502.21217, with the
dynamic/temporal detection results demonstrated in the gridworld module), or
an application paper analyzing a real dynamic system. At that point, add a
full `manuscript/` tree at the repo top level (config.yaml, section files
00–99, references.bib) following the docxology/template standard.
