from pathlib import Path

import pandas as pd
from pident.synthetic_data.obs_samplers import Timeseries


def write_monolix_csv(
    csv_path: Path, trajectories_per_patient: dict[str, dict[str, Timeseries]]
) -> None:
    """
    Writes the data in `trajectories_per_patient` to the csv at the given path,
    creating the parent directory if it doesn't already exist.

    Args:
        csv_path: Path to save the csv to
        trajectories_per_patient: dict mapping patient names to dicts that
            map observation variable names to observations
    """

    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    dict_for_pandas = {
        "time": [],
        "id": [],
        "observation": [],
        "observation_id": [],
        "observation_type": [],
    }

    # get all distinct obs var names; this is for converting obs var names
    # to obs ids
    # sort to ensure consistent order
    obs_vars = sorted(
        list(
            set(
                obs_var
                for trajectories in trajectories_per_patient.values()
                for obs_var in trajectories
            )
        )
    )

    for patient_name, trajectories in trajectories_per_patient.items():
        for obs_var_name, timeseries in trajectories.items():
            obs_id = obs_vars.index(obs_var_name)
            obs_times = timeseries["t"]
            observations = timeseries["y"]
            for time, obs in zip(obs_times, observations):  # type: ignore
                dict_for_pandas["time"].append(time)
                dict_for_pandas["id"].append(patient_name)
                dict_for_pandas["observation"].append(obs)
                dict_for_pandas["observation_id"].append(obs_id)
                dict_for_pandas["observation_type"].append(obs_var_name)

    pd.DataFrame(dict_for_pandas).to_csv(csv_path, index=False)
