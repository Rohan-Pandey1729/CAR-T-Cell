"""
pident: Practical Identifiability analysis for CAR-T cell models.

A toolkit for analyzing parameter identifiability in ODE models using synthetic data
generation, parameter estimation with Monolix, and performance metrics evaluation.

Main subpackages:
- common: Core utilities including distribution handling and ODE model wrappers
- estimation: Monolix configuration and estimation execution
- lib: Pre-built CAR-T cell models (Liu et al., Owens-Bozic)
- synthetic_data: Synthetic data generation and transformation for Monolix
- metrics: Analysis of estimation results and model performance
"""

from pident import common, estimation, lib, synthetic_data

__all__ = [
    "synthetic_data",
    "common",
    "estimation",
    "lib",
]
