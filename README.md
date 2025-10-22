# Lorenz System: Genetic Algorithm Optimization of Predictor-Corrector Methods

A university project exploring numerical integration methods for the Lorenz system, with a focus on optimizing predictor-corrector schemes using genetic algorithms.

> **Note:** This project was developed with the assistance of AI tools for implementation and analysis.

## Table of Contents
- [Overview](#overview)
- [The Lorenz System](#the-lorenz-system)
- [Methods Implemented](#methods-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)

## Overview

This project investigates the application of **Genetic Algorithms (GA)** to optimize numerical integration methods for solving the chaotic Lorenz system. The goal is to demonstrate that GA-based parameter optimization can improve the accuracy of predictor-corrector methods when compared against a high-accuracy baseline solution.

Four different approaches are implemented and compared:
1. **RK45 Baseline**: High-accuracy reference using Runge-Kutta 4/5
2. **ABM2 Predictor-Corrector**: Two-step Adams-Bashforth-Moulton method with Euler bootstrap
3. **GA-Optimized Step Size**: GA optimization of integration step size `h`
4. **GA-Optimized Coefficients**: GA optimization of both step size and predictor-corrector coefficients

## The Lorenz System

The Lorenz system is a set of three coupled ordinary differential equations that exhibit chaotic behavior:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

**System Parameters Used:**
- σ (sigma) = 10.0
- ρ (rho) = 28.0
- β (beta) = 8/3

**Initial Conditions:**
- x₀ = 1.0
- y₀ = 1.0
- z₀ = 1.0

**Integration Interval:**
- t ∈ [0, 50]

## Methods Implemented

### 1. RK45 Baseline (`lorenz_sys_RK45.py`)
Computes a high-accuracy reference solution using SciPy's `solve_ivp` with the Runge-Kutta 4/5 method. This serves as the "ground truth" for evaluating other methods.

### 2. Adams-Bashforth-Moulton Predictor-Corrector (`lorenz_pc.py`)
Implements a two-step Adams-Bashforth-Moulton (ABM2) scheme:
- **Predictor**: Adams-Bashforth 2-step (explicit)
- **Corrector**: Adams-Moulton 2-step (implicit)
- **Initialization**: Single Euler step for bootstrap
- **Fixed Step Size**: h = 0.01

### 3. GA-Optimized Step Size (`ga_optimized_pc_lorenz.py`)
Uses a genetic algorithm to find the optimal integration step size for the ABM2 method:
- **Population Size**: 12 individuals
- **Generations**: 20
- **Mutation Rate**: 0.25
- **Search Range**: h ∈ [0.0005, 0.015]
- **Fitness Function**: Mean squared error (MSE) against RK45 baseline

### 4. GA-Optimized Coefficients (`ga_multi_optimized_pc_lorenz.py`)
Extends GA optimization to both step size and predictor-corrector coefficients:
- **Parameters Optimized**: h, a₁, a₂, b₁, b₂
- **Population Size**: 30 individuals
- **Generations**: 40
- **Mutation Rate**: 0.3
- **Fitness Function**: MSE against RK45 baseline

## Project Structure

```
pc-ga-lorenz/
├── lorenz_sys_RK45.py              # RK45 baseline solution
├── lorenz_pc.py                     # ABM2 predictor-corrector
├── ga_optimized_pc_lorenz.py        # GA optimization of step size
├── ga_multi_optimized_pc_lorenz.py  # GA optimization of coefficients
├── lorenz_comparison_analysis.py    # Comparative analysis and visualization
├── requirements.txt                 # Project dependencies
├── experiments/
│   └── test.ipynb                   # Experimental notebook
└── results/
    ├── lorenz_baseline.npz          # RK45 solution data
    ├── lorenz_pc_abm2_euler.npz     # ABM2 solution data
    ├── lorenz_pc_ga_optimized.npz   # GA step-size optimized data
    ├── lorenz_pc_ga_multi.npz       # GA coefficients optimized data
    └── images/                      # Generated visualizations
        ├── lorenz_baseline_rk45.png
        ├── lorenz_baseline_rk45_timeseries.png
        ├── lorenz_ABM2_Euler.png
        ├── lorenz_ABM2_Euler_timeseries.png
        ├── lorenz_ga_optimized.png
        ├── lorenz_ga_optimized_timeseries.png
        ├── lorenz_multi_ga_optimized.png
        ├── lorenz_multi_ga_optimized_timeseries.png
        ├── convergence_ga_optimized.png
        ├── convergence_multi_ga_optimized.png
        └── lorenz_time_series_comparison.png
```

## Installation

### Prerequisites
- Python 3.11 or later
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pc-ga-lorenz
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in the following order to reproduce the complete analysis:

### 1. Generate Baseline Solution
```bash
python lorenz_sys_RK45.py
```
This creates the high-accuracy RK45 baseline and saves it to `results/lorenz_baseline.npz`.

### 2. Run Predictor-Corrector Method
```bash
python lorenz_pc.py
```
Generates the ABM2 solution with fixed step size.

### 3. Run GA Optimization (Step Size)
```bash
python ga_optimized_pc_lorenz.py
```
Optimizes the integration step size using a genetic algorithm. This may take several minutes.

### 4. Run GA Optimization (Coefficients)
```bash
python ga_multi_optimized_pc_lorenz.py
```
Optimizes both step size and predictor-corrector coefficients. This is the most computationally intensive step.

### 5. Generate Comparative Analysis
```bash
python lorenz_comparison_analysis.py
```
Produces comparative visualizations and quantitative metrics (RMSE, MAE) for all methods.

## Results

The comparative analysis produces several key outputs:

### Quantitative Metrics
- **RMSE (Root Mean Square Error)**: Measures overall deviation from baseline
- **MAE (Mean Absolute Error)**: Measures average absolute deviation
- **Relative Improvement**: Percentage improvement over standard ABM2 method

### Visualizations
- 3D trajectory plots of the Lorenz attractor for each method
- Time-series comparisons of x(t), y(t), z(t)
- GA convergence plots showing optimization progress
- Error evolution plots
- Comparative bar charts of RMSE and MAE

### Expected Outcomes
The GA-optimized methods typically demonstrate:
- Reduced error compared to fixed-step ABM2
- Improved stability in chaotic regions
- Better tracking of the RK45 baseline trajectory

## Technical Details

### Adams-Bashforth-Moulton Predictor-Corrector

The standard ABM2 formulas used:

**Predictor (Adams-Bashforth 2-step):**
```
y*_{n+1} = y_n + h/2 (3f_n - f_{n-1})
```

**Corrector (Adams-Moulton 2-step):**
```
y_{n+1} = y_n + h/2 (f*_{n+1} + f_n)
```

### Genetic Algorithm Parameters

**Step Size Optimization:**
- Encoding: Real-valued (h)
- Selection: Elitist (top 50%)
- Crossover: Linear interpolation (blend)
- Mutation: Gaussian perturbation

**Coefficient Optimization:**
- Encoding: Real-valued vector [h, a₁, a₂, b₁, b₂]
- Selection: Elitist (top 50%)
- Crossover: Linear interpolation
- Mutation: Gaussian perturbation with σ = 0.05

## Dependencies

Core dependencies (see `requirements.txt` for complete list):
- **NumPy**: Numerical computations and array operations
- **SciPy**: RK45 integration and scientific functions
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data analysis and metrics computation

## Academic Context

This project was developed as part of a university course to explore:
- Numerical methods for ordinary differential equations
- Optimization techniques in scientific computing
- Chaotic dynamical systems
- Genetic algorithms for parameter tuning

## License

This is an academic project developed for educational purposes.

---

**Acknowledgments:** This project utilized AI assistance for code development, optimization, and documentation.

