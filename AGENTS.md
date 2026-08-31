# AGENTS.md — `Markov_Blanket_Detection`

**What this is:** Real directory (not a symlink) in the ActiveInference lane,
hosting **DMBD — Dynamic Markov Blanket Detection**: a PyTorch framework for
detecting and analyzing Markov blankets in dynamic systems, extending static
Markov-blanket detection to temporal dependencies and their evolution over
time. Written from scratch (March 2025), inspired by
[pyDMBD](https://github.com/bayesianempirimancer/pyDMBD) and Beck & Ramstead,
*Dynamic Markov Blanket Detection for Macroscopic Physics Discovery*
([arXiv:2502.21217](https://arxiv.org/abs/2502.21217)).

Verified against disk 2026-08-31 (`ls`, repo `README.md`, `git remote -v`).

## Layout

- `DMBD/` — the Python package (detection and analysis code).
- `docs/` — documentation.
- `README.md` — overview, Markov-blanket structure, acknowledgments.
- `LICENSE.md`, `AGENTS.md` (this file).

## Gotchas

- Despite living under a `projects/ongoing/` lane mirror, **this directory is
  its own git repository** (remote
  `github.com/docxology/Markov_Blanket_Detection`, branch `main` — verified
  2026-08-31 with `git remote -v`). It is NOT covered by the parent
  checkout's gitignore; commit and push here directly. (Earlier note claiming
  "never committed" was wrong; superseded 2026-08-31.)
- Parent standard: see `../../AGENTS.md` (ActiveInference lane root).
- Underscore name (`Markov_Blanket_Detection`), unlike hyphenated siblings such
  as `cognitive-engine`; copy names exactly.
- Repo payload (source, tests, fixtures) is out of scope for doc passes; edit
  only this file and `README.md` when improving lane documentation.

## Agent quick reference

- **Backlog (canonical):** `TODO.md` — add findings there, one line each.
- **Run tests:** `python DMBD/run_tests.py` from repo root (requires
  `DMBD/.venv` or system torch/pandas; see README "Running Tests").
- **Gridworld demo:** `bash DMBD/run_gridworld_dmbd.sh` (see
  `DMBD/README_gridworld.md`).
- **Doc rules:** factual claims must trace to a file on disk; if
  unverifiable, write "Not documented in repo — needs owner input".
