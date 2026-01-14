"""
Functions for computing performance metrics comparing true vs estimated parameters.
"""

import numpy as np
from numpy.typing import ArrayLike


def compute_parameter_relative_error(
    true_values: ArrayLike, estimated_values: ArrayLike
) -> float:
    """
    Compute average relative error for a single parameter across individuals.

    Args:
        true_values: 1D array of true parameter values (one per individual)
        estimated_values: 1D array of estimated parameter values (same length)

    Returns:
        Mean absolute relative error: mean(|estimated - true| / |true|)

    Raises:
        ValueError: If arrays have different lengths or if any true value is 0
    """
    true_values = np.asarray(true_values)
    estimated_values = np.asarray(estimated_values)

    if true_values.shape != estimated_values.shape:
        raise ValueError(
            f"true_values and estimated_values must have same shape, "
            f"got {true_values.shape} and {estimated_values.shape}"
        )

    if np.any(true_values == 0):
        raise ValueError("Cannot compute relative error: true_values contains zeros")

    rel_errors = np.abs(estimated_values - true_values) / np.abs(true_values)
    return float(np.mean(rel_errors))


def compute_rms_linear(
    true_trajectories: ArrayLike, estimated_trajectories: ArrayLike
) -> float:
    """
    Compute RMS error between estimated and true trajectories in linear scale.

    Args:
        true_trajectories: 2D array of shape (n_vars, n_timepoints) with true trajectory values
        estimated_trajectories: 2D array of same shape with estimated trajectory values

    Returns:
        RMS error: sqrt(mean((estimated - true)^2))
    """
    true_traj = np.asarray(true_trajectories)
    est_traj = np.asarray(estimated_trajectories)

    if true_traj.shape != est_traj.shape:
        raise ValueError(
            f"Trajectories must have same shape, "
            f"got {true_traj.shape} and {est_traj.shape}"
        )

    squared_diffs = (est_traj - true_traj) ** 2
    return float(np.sqrt(np.mean(squared_diffs)))


def compute_rms_log10(
    true_trajectories: ArrayLike, estimated_trajectories: ArrayLike, eps: float = 1e-2
) -> float:
    """
    Compute RMS error between estimated and true trajectories in log10 scale.

    Args:
        true_trajectories: 2D array of shape (n_vars, n_timepoints) with true trajectory values
        estimated_trajectories: 2D array of same shape with estimated trajectory values
        eps: Small constant added before log to handle near-zero values (default: 1e-2)

    Returns:
        RMS error in log10 scale: sqrt(mean((log10(|estimated| + eps) - log10(|true| + eps))^2))
    """
    true_traj = np.asarray(true_trajectories)
    est_traj = np.asarray(estimated_trajectories)

    if true_traj.shape != est_traj.shape:
        raise ValueError(
            f"Trajectories must have same shape, "
            f"got {true_traj.shape} and {est_traj.shape}"
        )

    log_true = np.log10(np.abs(true_traj) + eps)
    log_est = np.log10(np.abs(est_traj) + eps)

    squared_diffs = (log_est - log_true) ** 2
    return float(np.sqrt(np.mean(squared_diffs)))


def compute_rms_per_variable(
    true_trajectories: ArrayLike,
    estimated_trajectories: ArrayLike,
    var_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute RMS error per observation variable in linear scale.

    Args:
        true_trajectories: 2D array of shape (n_vars, n_timepoints)
        estimated_trajectories: 2D array of same shape
        var_names: Optional list of variable names (default: use integer indices)

    Returns:
        Dict mapping variable name to RMS error for that variable
    """
    true_traj = np.asarray(true_trajectories)
    est_traj = np.asarray(estimated_trajectories)

    if true_traj.shape != est_traj.shape:
        raise ValueError("Trajectories must have same shape")

    n_vars = true_traj.shape[0]
    if var_names is None:
        var_names = [str(i) for i in range(n_vars)]

    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names length ({len(var_names)}) must match number of variables ({n_vars})"
        )

    result = {}
    for i, var_name in enumerate(var_names):
        squared_diffs = (est_traj[i] - true_traj[i]) ** 2
        result[var_name] = float(np.sqrt(np.mean(squared_diffs)))

    return result


def compute_rms_per_variable_log10(
    true_trajectories: ArrayLike,
    estimated_trajectories: ArrayLike,
    var_names: list[str] | None = None,
    eps: float = 1e-2,
) -> dict[str, float]:
    """
    Compute RMS error per observation variable in log10 scale.

    Args:
        true_trajectories: 2D array of shape (n_vars, n_timepoints)
        estimated_trajectories: 2D array of same shape
        var_names: Optional list of variable names (default: use integer indices)
        eps: Small constant added before log to handle near-zero values

    Returns:
        Dict mapping variable name to RMS error in log10 scale for that variable
    """
    true_traj = np.asarray(true_trajectories)
    est_traj = np.asarray(estimated_trajectories)

    if true_traj.shape != est_traj.shape:
        raise ValueError("Trajectories must have same shape")

    n_vars = true_traj.shape[0]
    if var_names is None:
        var_names = [str(i) for i in range(n_vars)]

    if len(var_names) != n_vars:
        raise ValueError(
            f"var_names length ({len(var_names)}) must match number of variables ({n_vars})"
        )

    log_true = np.log10(np.abs(true_traj) + eps)
    log_est = np.log10(np.abs(est_traj) + eps)

    result = {}
    for i, var_name in enumerate(var_names):
        squared_diffs = (log_est[i] - log_true[i]) ** 2
        result[var_name] = float(np.sqrt(np.mean(squared_diffs)))

    return result
