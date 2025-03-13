# Dynamic Markov Blanket Detection (DMBD)

A comprehensive framework for detecting and analyzing Markov blankets in dynamic systems using probabilistic techniques based on PyTorch.

## Overview

Dynamic Markov Blanket Detection (DMBD) provides tools for identifying causal relationships in complex systems by detecting and analyzing Markov blankets. A Markov blanket for a node in a graph consists of its parents, children, and the parents of its children (spouses), and shields the node from the rest of the network - making the node conditionally independent of all other variables given its Markov blanket.

This framework extends traditional static Markov blanket detection to dynamic systems, allowing for the detection of temporal dependencies and the evolution of Markov blankets over time.

## Acknowledgments

- The framework builds on concepts from causal inference, information theory, and active inference frameworks
- This repository was written from scratch using Cursor and Claude 3.7 on March 13, 2025, inspired by:
  - [pyDMBD](https://github.com/bayesianempirimancer/pyDMBD)
  - [Dynamic Markov Blanket Detection for Macroscopic Physics Discovery](https://arxiv.org/abs/2502.21217) (Beck & Ramstead, 2025)


### Markov Blanket Structure

```mermaid
graph LR
    subgraph "Markov Blanket"
        P[Parents] --> T[Target Node]
        T --> C[Children]
        P -.-> S[Spouses]
        S -.-> C
    end
    O[Other Nodes] -.-> P
    C -.-> E[External Nodes]
    
    classDef target fill:#f96, stroke:#333, stroke-width:2px
    classDef blanket fill:#bbf, stroke:#333, stroke-width:1px
    classDef external fill:#ececec, stroke:#333, stroke-width:1px
    
    class T target
    class P,C,S blanket
    class O,E external
```

### Dynamic Markov Blanket 

```mermaid
graph TB
    subgraph t1["Time t-1"]
        P1[Parents t-1] --> T1[Target t-1]
        T1 --> C1[Children t-1]
    end
    
    subgraph t2["Time t"]
        P2[Parents t] --> T2[Target t]
        T2 --> C2[Children t]
    end
    
    subgraph t3["Time t+1"]
        P3[Parents t+1] --> T3[Target t+1]
        T3 --> C3[Children t+1]
    end
    
    T1 --> T2 --> T3
    C1 -.-> P2
    C2 -.-> P3
    
    classDef timeframe fill:#ececec, stroke:#333, stroke-width:1px
    classDef target fill:#f96, stroke:#333, stroke-width:2px
    classDef blanket fill:#bbf, stroke:#333, stroke-width:1px
    
    class t1,t2,t3 timeframe
    class T1,T2,T3 target
    class P1,C1,P2,C2,P3,C3 blanket
```

## Features

- **Markov Blanket Detection**: Identify the Markov blanket of target variables in static data
- **Dynamic Markov Blanket Detection**: Detect time-varying Markov blankets in temporal data
- **Cognitive Structure Identification**: Identify cognitive structures (sensory, action, internal states) within Markov blankets
- **Data Partitioning**: Tools for partitioning data based on Markov blanket components
- **Visualization**: Rich visualization tools for Markov blankets, cognitive structures, and information flow
- **Comprehensive Interface**: High-level interface for analysis combining all components
- **PyTorch Integration**: Built on PyTorch for GPU acceleration and deep learning compatibility

### DMBD Analysis Workflow

```mermaid
flowchart TD
    subgraph Inputs
        A[Raw Data] --> B[Preprocessed Data]
        C[Target Variable Selection]
    end
    
    subgraph Analysis
        B --> D[Static Markov Blanket Detection]
        B --> E[Dynamic Markov Blanket Detection]
        C --> D
        C --> E
        D --> F[Cognitive Structure Identification]
        E --> F
        F --> G[Data Partitioning]
    end
    
    subgraph Outputs
        G --> H[Visualization]
        G --> I[Statistical Analysis]
        G --> J[Modeling]
    end
    
    classDef input fill:#d1f0a9, stroke:#333, stroke-width:1px
    classDef process fill:#bbdefb, stroke:#333, stroke-width:1px
    classDef output fill:#ffccbc, stroke:#333, stroke-width:1px
    
    class A,B,C input
    class D,E,F,G process
    class H,I,J output
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+

### Using pip

```bash
pip install dmbd
```

### Development Installation

```bash
git clone https://github.com/openmanus/dmbd.git
cd dmbd
pip install -e .
```

## Quick Start

```python
import pandas as pd
import numpy as np
from dmbd.src.dmbd_analyzer import DMBDAnalyzer

# Load data
data = pd.read_csv('your_data.csv')

# Initialize analyzer
analyzer = DMBDAnalyzer(data)

# Analyze target variable
target_idx = 0  # Index of the target variable
results = analyzer.analyze_target(target_idx)

# Print basic results
print(f"Markov Blanket for X{target_idx}:")
print(f"  Parents: {results['markov_blanket']['parents']}")
print(f"  Children: {results['markov_blanket']['children']}")
print(f"  Spouses: {results['markov_blanket']['spouses']}")
print(f"  Blanket Size: {results['markov_blanket']['blanket_size']}")

# Visualize Markov blanket
analyzer.visualize_markov_blanket(target_idx)

# Visualize cognitive structures
analyzer.visualize_cognitive_structures(target_idx)
```

## Synthetic Data Example

The framework includes tools for generating synthetic data for testing:

```python
from dmbd.src.examples import generate_synthetic_data, basic_markov_blanket_example

# Generate synthetic data
data = generate_synthetic_data(
    n_samples=500,
    n_vars=8,
    causal_density=0.3,
    temporal=True,
    n_time_points=5
)

# Or run the built-in example
basic_markov_blanket_example()
```

## Framework Components

### Component Relationships

```mermaid
classDiagram
    DMBDAnalyzer --> MarkovBlanket
    DMBDAnalyzer --> DynamicMarkovBlanket
    DMBDAnalyzer --> CognitiveIdentification
    DMBDAnalyzer --> DataPartitioning
    DMBDAnalyzer --> MarkovBlanketVisualizer
    
    class DMBDAnalyzer {
        +data
        +analyze_target()
        +visualize_markov_blanket()
        +visualize_cognitive_structures()
    }
    
    class MarkovBlanket {
        +data
        +detect_blanket()
        +classify_nodes()
    }
    
    class DynamicMarkovBlanket {
        +data
        +time_column
        +lag
        +detect_dynamic_blanket()
    }
    
    class CognitiveIdentification {
        +data
        +markov_blanket
        +identify_cognitive_structures()
    }
    
    class DataPartitioning {
        +data
        +partition_data()
    }
    
    class MarkovBlanketVisualizer {
        +plot_markov_blanket()
        +plot_cognitive_structures()
        +plot_information_flow()
    }
```

### MarkovBlanket

Detects Markov blankets in static data using conditional independence tests.

```python
from framework.markov_blanket import MarkovBlanket

mb = MarkovBlanket(data)
parents, children, spouses = mb.detect_blanket(target_idx)
```

```mermaid
flowchart LR
    subgraph Independence Tests
        CI[Conditional Independence Testing]
    end
    
    subgraph Blanket Detection
        A[Find Parents] --> D[Detect Markov Blanket]
        B[Find Children] --> D
        C[Find Spouses] --> D
    end
    
    CI --> A
    CI --> B
    CI --> C
    
    D --> E[Parents]
    D --> F[Children]
    D --> G[Spouses]
    
    classDef methods fill:#e1bee7, stroke:#333, stroke-width:1px
    classDef outputs fill:#bbdefb, stroke:#333, stroke-width:1px
    
    class CI,A,B,C,D methods
    class E,F,G outputs
```

### DynamicMarkovBlanket

Extends Markov blanket detection to temporal data.

```python
from framework.markov_blanket import DynamicMarkovBlanket

dmb = DynamicMarkovBlanket(data, time_column='time', lag=2)
dynamic_blanket = dmb.detect_dynamic_blanket(target_idx)
```

```mermaid
sequenceDiagram
    participant T as Time Series Data
    participant DMB as DynamicMarkovBlanket
    participant W as Window Generator
    participant MB as MarkovBlanket
    
    T->>DMB: Initialize with data
    DMB->>W: Generate time windows
    loop For each window
        W->>MB: Create dataset slice
        MB->>MB: Detect static blanket
    end
    MB->>DMB: Aggregate results
    DMB->>DMB: Analyze temporal consistency
    DMB->>DMB: Build dynamic blanket
```

### CognitiveIdentification

Identifies cognitive structures within Markov blankets.

```python
from framework.cognitive_identification import CognitiveIdentification

ci = CognitiveIdentification(data, mb)
structures = ci.identify_cognitive_structures(target_idx)
```

```mermaid
graph TD
    subgraph Cognitive Structures
        S[Sensory States]
        A[Active States]
        I[Internal States]
    end
    
    subgraph Markov Blanket
        P[Parents] --> T[Target]
        T --> C[Children]
        P -.-> SP[Spouses]
        SP -.-> C
    end
    
    P --> |Information Flow Analysis| S
    C --> |Information Flow Analysis| A
    T --> |State Classification| I
    
    classDef cognitive fill:#ffcc80, stroke:#333, stroke-width:1px
    classDef blanket fill:#bbdefb, stroke:#333, stroke-width:1px
    classDef target fill:#f48fb1, stroke:#333, stroke-width:2px
    
    class S,A,I cognitive
    class P,C,SP blanket
    class T target
```

### DataPartitioning

Tools for partitioning data based on Markov blanket components.

```python
from framework.data_partitioning import DataPartitioning

dp = DataPartitioning(data)
classifications = mb.classify_nodes(target_idx)
partitions = dp.partition_data(classifications)
```

```mermaid
pie
    title "Example Data Partition"
    "Target" : 1
    "Markov Blanket" : 12
    "External" : 87
```

### MarkovBlanketVisualizer

Visualization tools for Markov blankets and cognitive structures.

```python
from framework.visualization import MarkovBlanketVisualizer

visualizer = MarkovBlanketVisualizer()
fig = visualizer.plot_markov_blanket(mb, target_idx)
```

```mermaid
graph TD
    subgraph Visualization Functions
        MB[plot_markov_blanket]
        CS[plot_cognitive_structures]
        IF[plot_information_flow]
        TS[plot_time_series]
        NC[plot_network_centrality]
    end
    
    subgraph Outputs
        PNG[Static Images]
        HTML[Interactive Plots]
        MP4[Animations]
    end
    
    MB --> PNG
    MB --> HTML
    CS --> PNG
    CS --> HTML
    IF --> HTML
    IF --> MP4
    TS --> HTML
    NC --> HTML
    
    classDef functions fill:#c5e1a5, stroke:#333, stroke-width:1px
    classDef outputs fill:#ffcc80, stroke:#333, stroke-width:1px
    
    class MB,CS,IF,TS,NC functions
    class PNG,HTML,MP4 outputs
```

## Running Tests

To run the test suite:

```bash
python -m unittest discover tests
```

Or to run a specific test:

```bash
python -m unittest tests.test_markov_blanket
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE.md](LICENSE.md) file for details.
