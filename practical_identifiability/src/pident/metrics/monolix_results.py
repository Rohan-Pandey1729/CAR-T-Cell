"""
Functions for extracting data from Monolix estimation results.

Note: These functions have only been tested with Monolix version 2024R1.
Output formats from other versions may not be compatible.
"""

from pathlib import Path
from typing import NamedTuple

from pident.common.models import ODEModel


class EstimatedParametersResult(NamedTuple):
    """Result of parsing estimated parameters from Monolix output."""

    patient_ids: list[str]
    param_dicts: list[dict[str, dict[str, float]]]


def parse_convergence_results(results_dir: Path) -> dict[str, bool | None]:
    """
    Extract convergence flags from Monolix results directory.

    Args:
        results_dir: Path to the Monolix results directory (typically named after the .mlxtran file)

    Returns:
        Dict with keys "exploratory_converged" and "smoothing_converged", each with value
        True (converged), False (did not converge), or None (could not parse).

    Note:
        Looks for "Autostop" keyword in the summary.txt file. Returns None if the file
        cannot be found or parsed.
    """
    summary_fp = results_dir / "summary.txt"

    convergence: dict[str, bool | None] = {
        "exploratory_converged": None,
        "smoothing_converged": None,
    }

    if not summary_fp.exists():
        return convergence

    try:
        with open(summary_fp, "r") as f:
            for line in f:
                if "Exploratory phase" in line:
                    convergence["exploratory_converged"] = "Autostop" in line
                elif "Smoothing phase" in line:
                    convergence["smoothing_converged"] = "Autostop" in line
    except Exception:
        pass

    return convergence


def parse_estimated_parameters(
    results_dir: Path, ode_model: ODEModel
) -> EstimatedParametersResult:
    """
    Extract estimated parameters from Monolix estimatedIndividualParameters.txt.

    Args:
        results_dir: Path to the Monolix results directory
        ode_model: ODEModel instance to validate expected parameters and initial conditions

    Returns:
        EstimatedParametersResult with:
        - patient_ids: list of patient ID strings in order from the file
        - param_dicts: list of dicts where param_dicts[i] contains:
            {
                "param_name": {"SAEM": float, "mean": float, "mode": float, "sd": float},
                ...
            }
            Only statistics that are present in the file are included.

    Raises:
        FileNotFoundError: If estimatedIndividualParameters.txt does not exist
        ValueError: If expected parameters/initial conditions from ode_model are missing
        ValueError: If the file format is unexpected (e.g., no header row)

    Note:
        Assumes columns follow the pattern: {name}_{statistic} where name is a parameter
        or initial condition name (e.g., "a_mode", "T0_SAEM") and statistic is one of
        "SAEM", "mean", "mode", "sd".

        This function validates that all parameters in ode_model.param_names and all
        initial conditions (inferred as obs_var_names with "0" suffix) are present.
    """
    param_file = results_dir / "estimatedIndividualParameters.txt"

    if not param_file.exists():
        raise FileNotFoundError(
            f"estimatedIndividualParameters.txt not found in {results_dir}"
        )

    # Expected parameter/initial condition names
    expected_names = set(ode_model.param_names) | {
        f"{obs_var}0" for obs_var in ode_model.obs_var_names
    }

    patient_ids = []
    param_dicts = []

    with open(param_file, "r") as f:
        # Parse header
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError("estimatedIndividualParameters.txt has no header row")

        headers = header_line.split(",")

        # Parse header to identify columns
        # Columns are like "a_SAEM", "T0_mean", etc.
        # Build a mapping: param_name -> {statistic: column_index}
        col_map = {}  # {param_name: {statistic: column_index}}

        for col_idx, col_name in enumerate(headers):
            # Split on rightmost underscore
            parts = col_name.rsplit("_", 1)
            if len(parts) == 2:
                param_name, statistic = parts
                if param_name not in col_map:
                    col_map[param_name] = {}
                col_map[param_name][statistic] = col_idx

        # Validate that all expected parameters/initial conditions are present
        found_names = set(col_map.keys())
        missing = expected_names - found_names
        if missing:
            raise ValueError(
                f"Missing parameters/initial conditions in estimatedIndividualParameters.txt: {missing}"
            )

        # Parse data rows
        for line in f:
            line = line.strip()
            if not line:
                continue

            values = line.split(",")
            patient_id = values[0]
            patient_ids.append(patient_id)

            param_dict = {}
            for param_name in expected_names:
                param_dict[param_name] = {}
                for stat, col_idx in col_map[param_name].items():
                    try:
                        param_dict[param_name][stat] = float(values[col_idx])
                    except (ValueError, IndexError):
                        # Leave missing or unparseable values out
                        pass

            param_dicts.append(param_dict)

    return EstimatedParametersResult(patient_ids, param_dicts)
