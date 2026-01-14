"""
Pre-built CAR-T cell models and analysis utilities.

Submodules:
- liu_et_al_model: Liu et al. CAR-T cell dynamics model (with and without nN variable)
- owens_bozic_model: Owens-Bozic CAR-T cell dynamics model
- metrics: Functions for computing parameter and trajectory performance metrics
- stratification: Functions for stratifying individuals by outcomes
"""

from pident.lib import liu_et_al_model, metrics, owens_bozic_model, stratification

__all__ = [
    "liu_et_al_model",
    "metrics",
    "owens_bozic_model",
    "stratification",
]
