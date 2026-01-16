# pident: Practical Identifiability Analysis for ODE Models

A Python toolkit for analyzing **practical identifiability** in ODE-based mathematical models through synthetic data generation, parameter estimation, and performance metrics evaluation.

## Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- Monolix 2024R1 (for parameter estimation; path must be set in `.env`)

### Setup (Recommended: Using Conda)

1. **Create the conda environment** from the provided `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate cartcell
   ```

2. **Configure Monolix path** in `practical_identifiability/src/pident/.env`:

   ```
   MONOLIX_PATH=/path/to/Monolix
   ```

   (see Monolix documentation if you are unsure where this path is on your machine).

   You can copy `src/pident/.env.example` as a template:

   ```bash
   cp practical_identifiability/src/pident/.env.example practical_identifiability/src/pident/.env
   ```

### Alternative: Using pip with venv

If you prefer not to use conda:

1. **Navigate to the project directory**:

   ```bash
   cd practical_identifiability
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package with dependencies**:

   ```bash
   pip install -e .
   ```

4. **Configure Monolix path** in `src/pident/.env` as described above.

## Overview

Practical identifiability assesses whether model parameters can be uniquely determined from experimental data given realistic noise levels and measurement schedules. This package provides an end-to-end workflow for identifiability studies:

1. **Synthetic Data Generation**: Sample parameters from prior distributions, simulate ODE trajectories, and add realistic observation noise
2. **Parameter Estimation**: Configure and run Monolix parameter estimation on synthetic data
3. **Metrics Evaluation**: Parse estimation results and compute parameter recovery and trajectory prediction metrics
4. **Outcome Stratification**: Assess identifiability across clinical outcomes

More specifically, various modules implement the following features (see docstrings for more detail):

1. **Parameter Sampling**: Draw parameter values from prior distributions (e.g., lognormal for positive parameters)
2. **Trajectory Simulation**: Solve ODE system to generate noiseless ground truth
3. **Observation Sampling**: Subsample trajectories at realistic measurement timepoints
4. **Noise Addition**: Add measurement error following realistic error models (e.g., proportional normal error)
5. **Monolix Estimation**: Fit parameters to synthetic data using population parameter estimation
6. **Metrics Evaluation**: Compute parameter recovery accuracy and prediction error

## Usage Examples

For a complete example, see [notebooks/ob_pipeline_test.ipynb](notebooks/ob_pipeline_test.ipynb).

## Extending the package

### Adding a new model

1. Create a new module in `pident/lib/` with ODE function and parameter/variable name lists
2. Instantiate an `ODEModel` object
3. Use the same interface as existing models (`get_ground_truth`, etc.)
