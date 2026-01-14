"""
Functions for stratifying individuals by outcomes or other criteria.
"""

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

StratificationFunc = Callable[[list], list[str]]
"""Type for a function that takes a list of ground truth values/trajectories and returns outcome labels."""


def stratify_by_final_value(
    ground_truth_values: list[ArrayLike],
    threshold: float,
    var_index: int = 0,
) -> list[str]:
    """
    Stratify individuals into two groups based on final value of a variable.

    Args:
        ground_truth_values: List of values/arrays, one per individual. Can be:
            - Scalar values (float, np.float64, etc.)
            - 1D arrays containing values at different timepoints for a single variable
            - Multi-dimensional arrays where var_index selects the variable
        threshold: Threshold value for classification
        var_index: Index of the variable to use (default: 0, for use when values
            contain multiple variables)

    Returns:
        List of outcome labels ("above_threshold" or "below_threshold"), one per individual

    Example:
        For tumor size with threshold=1000:
        - If tumor_size[-1] > 1000 -> "above_threshold" (NR)
        - Else -> "below_threshold" (CR)
    """
    outcomes = []
    for values in ground_truth_values:
        values_arr = np.asarray(values)
        # Handle scalar values (0-dimensional arrays)
        if values_arr.ndim == 0:
            final_value = float(values_arr)
        # Handle 1D arrays
        elif values_arr.ndim == 1:
            final_value = values_arr[-1]
        # Handle higher-dimensional arrays
        else:
            final_value = values_arr[var_index, -1]

        if final_value > threshold:
            outcomes.append("above_threshold")
        else:
            outcomes.append("below_threshold")

    return outcomes


def stratify_by_custom_function(
    ground_truth_values: list[ArrayLike],
    classify_fn: Callable[[ArrayLike], str],
) -> list[str]:
    """
    Stratify individuals using a custom classification function.

    Args:
        ground_truth_values: List of arrays (trajectories or values), one per individual
        classify_fn: Function that takes a single individual's values/trajectory and
            returns an outcome label string

    Returns:
        List of outcome labels, one per individual
    """
    return [classify_fn(values) for values in ground_truth_values]


def filter_by_outcome(
    data: list,
    outcomes: list[str],
    target_outcome: str,
) -> list:
    """
    Filter a list of data items by outcome group.

    Args:
        data: List of data items (parameters, trajectories, etc.)
        outcomes: List of outcome labels, one per data item (must be same length as data)
        target_outcome: The outcome label to filter for

    Returns:
        List containing only items from data where outcomes[i] == target_outcome

    Raises:
        ValueError: If data and outcomes have different lengths
    """
    if len(data) != len(outcomes):
        raise ValueError(
            f"data and outcomes must have same length, got {len(data)} and {len(outcomes)}"
        )

    return [item for item, outcome in zip(data, outcomes) if outcome == target_outcome]
