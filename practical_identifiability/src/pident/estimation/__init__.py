"""
Monolix configuration and estimation execution.

Submodules:
- mlx_config: Configuration dataclasses and Monolix template generation
- mlx_estimation: Execution of Monolix parameter estimation runs
"""

from pident.estimation import mlx_config, mlx_estimation

__all__ = [
    "mlx_config",
    "mlx_estimation",
]
