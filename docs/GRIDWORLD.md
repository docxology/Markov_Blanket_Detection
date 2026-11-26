# Synthetic gridworld

The [simulator](../DMBD/src/gridworld_simulation.py) produces a time series
of cell activations from a Gaussian profile moving around a circle.
The [gridworld analyzer](../DMBD/src/gridworld_dmbd.py) turns those
activations into heuristic partition scores, images, and animations.

## Generate a disposable analysis

Follow [local setup](USAGE.md) and ensure ffmpeg is available. Run from the
repository root using the active environment.

```bash
gridworld_output="$(mktemp -d)"
MPLBACKEND=Agg OMP_NUM_THREADS=1 bash DMBD/run_gridworld_dmbd.sh \
  --grid-size 5 5 --time-points 3 --radius 2 --sigma 0.5 \
  --output-dir "$gridworld_output"
printf '%s\n' "$gridworld_output/report.html"
```

The wrapper runs [simulation and analysis](../DMBD/run_gridworld_analysis.py)
and then builds [the HTML report](../DMBD/generate_gridworld_report.py). Use
`--help` for the current flags. The wrapper defaults to PyTorch. The
standalone analysis command enables that path when passed `--use-torch`.
The wrapper’s `--include-raw-data` option adds raw data to the report.

A run can produce simulation CSV, image and video files under
`simulation/`, partition images and animations under `analysis/`, and
`report.html`. The [historical runs](../DMBD/output/) retain earlier
parameter choices, visualizations, and reports. Use a fresh output
directory for comparisons.

The moving profile is a synthetic numerical process with known
geometry. Its partition visualizations describe the algorithm's
assignments under selected settings.

## Background

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*.
- Friston, K. J. (2013). *Life as we know it*. Journal of the Royal Society Interface, 10(86), 20130475.
- Kirchhoff, M., Parr, T., Palacios, E., Friston, K., and Kiverstein, J. (2018).
  *The Markov blankets of life: autonomy, active inference and the free energy principle*.
  Journal of the Royal Society Interface, 15(138), 20170792.
