"""
Synthetic data generation and transformation for identifiability studies.

Submodules:
- dist_samplers: Multivariate distribution sampling utilities
- obs_samplers: Trajectory subsampling from ODE model solutions
- obs_noise: Adding realistic observation noise to trajectories
- monolix_csv_writer: Writing synthetic data to Monolix-compatible CSV format
"""

from pident.synthetic_data import (
    dist_samplers,
    monolix_csv_writer,
    obs_noise,
    obs_samplers,
)

__all__ = ["dist_samplers", "monolix_csv_writer", "obs_noise", "obs_samplers"]
