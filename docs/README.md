# Markov_Blanket_Detection (DMBD) documentation

Dynamic Markov Blanket Detection: a PyTorch framework for detecting and
analyzing Markov blankets in dynamic systems (parents, children, spouses;
temporal evolution of blankets). See root `README.md` for the full overview,
mermaid blanket-structure diagram, and acknowledgments (built from scratch
March 13, 2025, inspired by pyDMBD and Beck & Ramstead 2025,
arXiv:2502.21217).

## Repo layout

All implementation lives in the `DMBD/` package directory:

| Path | Contents |
|---|---|
| `DMBD/framework/` | Core framework modules (markov_blanket, data_partitioning, cognitive_identification, visualization) |
| `DMBD/src/`, `DMBD/tests/` | Source and test code |
| `DMBD/run_tests.py`, `DMBD/run_dmbd_tests.sh` | Test entry points |
| `DMBD/run_gridworld_*.py|.sh`, `DMBD/README_gridworld.md` | Gridworld analysis (moving Gaussian blur demo) and its report generator |
| `DMBD/generate_report.py`, `DMBD/analyze_partition.py`, `DMBD/direct_visualization.py` | Reporting and visualization utilities |
| `DMBD/pyproject.toml`, `DMBD/setup.py`, `DMBD/install_dependencies.sh` | Packaging (project name `dmbd`, deps: torch, numpy, pandas, scikit-learn, matplotlib, tqdm, networkx, seaborn) |
| `DMBD/output/` | Generated analysis outputs |

## How to run/test

```bash
# install dependencies (see DMBD/install_dependencies.sh or DMBD/pyproject.toml)
python DMBD/run_tests.py            # test suite
bash DMBD/run_dmbd_tests.sh         # shell test wrapper
python DMBD/run_gridworld_test.py   # gridworld demo test
bash DMBD/run_gridworld_dmbd.sh     # full gridworld analysis
```

## Documentation here

This `docs/` folder is intentionally minimal: root `README.md` is the main
entry point and `DMBD/README_gridworld.md` documents the gridworld module.
Add one file per real concern here only when a concern outgrows those.
