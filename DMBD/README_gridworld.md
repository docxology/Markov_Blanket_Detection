# Gridworld Dynamic Markov Blanket Analysis

This module extends the Dynamic Markov Blanket Detection (DMBD) framework to analyze gridworld simulations with moving Gaussian blurs. It provides tools for detecting Dynamic Markov Blankets and visualizing their evolution over time.

## Overview

The gridworld simulation creates a 2D grid environment where a Gaussian blur moves in a circular path. This creates dynamic patterns of activation across the grid cells, which can be analyzed to detect Dynamic Markov Blankets and cognitive structures.

The module provides:

- A gridworld simulation with a moving Gaussian blur
- Methods for detecting Dynamic Markov Blankets in the simulation data
- Visualizations of the evolving Markov blanket partitions
- Animations of the evolving cognitive structures

## Usage

### Quick Start

To run a complete analysis with default parameters, simply execute:

```bash
./run_gridworld_dmbd.sh
```

This will:
1. Generate a gridworld simulation with a moving Gaussian blur
2. Run the Dynamic Markov Blanket analysis
3. Create visualizations and animations
4. Generate an HTML report with the results

### Command-Line Options

For more control, you can specify various options:

```bash
./run_gridworld_dmbd.sh --grid-size 20 20 --time-points 50 --radius 8.0 --sigma 1.5 --threshold 0.1 --output-dir output/custom_analysis
```

Options:
- `--grid-size HEIGHT WIDTH`: Size of the grid (default: 30 30)
- `--time-points NUM`: Number of time points to simulate (default: 100)
- `--radius NUM`: Radius of the circular path (default: 10.0)
- `--sigma NUM`: Standard deviation of the Gaussian blur (default: 2.0)
- `--threshold NUM`: Threshold for Markov blanket detection (default: 0.1)
- `--output-dir DIR`: Directory to save outputs (default: output/gridworld_dmbd)
- `--use-torch`: Use PyTorch for computations (default: true)
- `--skip-animation`: Skip generating animations (default: false)
- `--quick-mode`: Run a quicker analysis with smaller grid (default: false)
- `--include-raw-data`: Include raw data in the HTML report (default: false)

### Individual Components

You can also run the individual components separately:

```bash
# Generate gridworld simulation and analyze it
python3 run_gridworld_analysis.py --grid-size 30 30 --time-points 100 --output-dir output/analysis_only

# Generate HTML report from existing analysis
python3 generate_gridworld_report.py --analysis-dir output/analysis_only --output-file output/report.html
```

## Files

- `run_gridworld_dmbd.sh`: Main script to run the complete analysis pipeline
- `run_gridworld_analysis.py`: Script to run the gridworld simulation and DMBD analysis
- `generate_gridworld_report.py`: Script to generate HTML reports from analysis results
- `src/gridworld_simulation.py`: Module for gridworld simulation with a moving Gaussian blur
- `src/gridworld_dmbd.py`: Module for analyzing gridworld data using DMBD techniques

## Requirements

- Python 3.6+
- NumPy
- Pandas
- PyTorch
- Matplotlib
- tqdm

## Output

The analysis generates several outputs:

1. **Simulation data**:
   - CSV file with time series data for each grid cell
   - Animation of the moving Gaussian blur
   - Example frame images

2. **Markov Blanket analysis**:
   - Static visualizations of Markov blanket partitions
   - Animations of evolving Markov blankets
   - Cognitive structure visualizations

3. **HTML Report**:
   - Summary of the analysis parameters
   - Embedded visualizations and animations
   - Interpretation of the results

## Interpretation

The Dynamic Markov Blanket analysis of the gridworld simulation demonstrates how causal relationships and information flow can be inferred from time-series data. The moving Gaussian blur creates patterns of activation that propagate across the grid, allowing us to study how information flows spatially and temporally.

Key observations:
- The Markov Blanket of a cell typically includes its spatial neighbors
- As the Gaussian blur moves, Markov Blankets adapt to track this movement
- The cognitive partition identifies sensory, action, and internal state nodes
- The analysis reveals self-organizing cognitive structures without explicit design

## References

For more information on Dynamic Markov Blankets and their applications, see:
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
- Friston, K. J. (2013). Life as we know it. Journal of the Royal Society Interface, 10(86), 20130475.
- Kirchhoff, M., Parr, T., Palacios, E., Friston, K., & Kiverstein, J. (2018). The Markov blankets of life: autonomy, active inference and the free energy principle. Journal of the Royal Society Interface, 15(138), 20170792. 