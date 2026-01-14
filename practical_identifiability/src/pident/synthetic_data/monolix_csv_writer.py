"""
Write synthetic trajectories to Monolix-compatible CSV format.

Provides functions to transform synthetic data (ODE solutions with noise)
into the standardized CSV format required by Monolix for parameter estimation.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pident.common.models import ODEModel
from pident.synthetic_data.obs_samplers import Timeseries


def write_monolix_csv(
    csv_path: Path,
    trajectories_per_patient: dict[str, dict[str, Timeseries]],
    ode_model: ODEModel,
    transformations: dict[str, Callable[[ArrayLike], ArrayLike]] | None = None,
    hidden_obs_var_names: list[str] | None = None,
) -> None:
    """
    Writes the data in `trajectories_per_patient` to the csv at the given path,
    creating the parent directory if it doesn't already exist.

    Args:
        csv_path: Path to save the csv to
        trajectories_per_patient: dict mapping patient names to dicts that
            map observation variable names to observations
        ode_model: ODEModel for validation of observation variable names.
            All observation variables must be in ode_model.obs_var_names.
        transformations: Optional dict mapping observation variable names to transformation
            functions. Each function takes an ArrayLike of observations and returns transformed
            observations. If provided, all keys must match observation variables in ode_model.
        hidden_obs_var_names: Optional list of observation variable names to exclude from the csv.
            All names must be valid observation variables in ode_model.obs_var_names.

    Raises:
        ValueError: If observation variables don't match ode_model,
            if transformations has invalid variable names,
            or if hidden_obs_var_names contains invalid variable names.
    """

    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate observation variables against ode_model
    obs_var_names_all = set(
        obs_var
        for trajectories in trajectories_per_patient.values()
        for obs_var in trajectories
    )
    invalid_obs_vars = obs_var_names_all - set(ode_model.obs_var_names)
    if invalid_obs_vars:
        raise ValueError(f"Observation variables not in ode_model: {invalid_obs_vars}")

    # Validate transformations if provided
    if transformations is not None:
        invalid_transforms = set(transformations.keys()) - set(ode_model.obs_var_names)
        if invalid_transforms:
            raise ValueError(
                f"Transformations contain invalid observation variable names: {invalid_transforms}"
            )

    # Validate and normalize hidden_obs_var_names
    if hidden_obs_var_names is None:
        hidden_obs_var_names = []
    invalid_hidden = set(hidden_obs_var_names) - set(ode_model.obs_var_names)
    if invalid_hidden:
        raise ValueError(
            f"hidden_obs_var_names contains invalid observation variable names: {invalid_hidden}"
        )

    dict_for_pandas = {
        "time": [],
        "id": [],
        "observation": [],
        "observation_id": [],
        "observation_type": [],
    }

    for patient_name, trajectories in trajectories_per_patient.items():
        for obs_var_name, timeseries in trajectories.items():
            # Skip hidden observation variables
            if obs_var_name in hidden_obs_var_names:
                continue

            obs_id = ode_model.obs_var_names.index(obs_var_name)
            obs_times = timeseries["t"]
            observations = np.array(timeseries["y"])

            # Apply transformation if provided
            if transformations is not None and obs_var_name in transformations:
                observations = transformations[obs_var_name](observations)

            for time, obs in zip(obs_times, observations):  # type: ignore
                dict_for_pandas["time"].append(time)
                dict_for_pandas["id"].append(patient_name)
                dict_for_pandas["observation"].append(obs)
                dict_for_pandas["observation_id"].append(obs_id)
                dict_for_pandas["observation_type"].append(obs_var_name)

    pd.DataFrame(dict_for_pandas).to_csv(csv_path, index=False)
