from typing import Callable

import numpy as np
from numpy.random import Generator
from pident.common.distributions import TruncNorm
from pident.common.exceptions import DomainError
from pident.synthetic_data.obs_samplers import Timeseries

ErrorModel = Callable[[float, Generator], float]


def add_noise(
    trajectories: dict[str, Timeseries],
    error_models: dict[str, ErrorModel],
    rng: Generator,
) -> dict[str, Timeseries]:
    """
    Transforms the given `trajectories` by observation variable,
    assumed to be without noise, into noisy trajectories according to
    an error model per observation variable.

    `trajectories` should be a dict from observation variable names to
    dicts, each with a key "t" for the times at which observations
    were collected for that variable and a key "y" for the observation values
    at those times.

    `error_models` should be a dict from observation variable names to
    functions which accept a scalar and a `numpy.random.Generator` and
    return a scalar.

    The return value should be of the same form as `trajectories`
    but with every observation ndarray ("y" value) replaced with
    an ndarray with noisy observations.
    "t" values are copies of the ones in `trajectories`.
    """
    res: dict[str, Timeseries] = {}
    for obs_var, trajectory in trajectories.items():
        y_vals = np.array(trajectory["y"])
        error_model = error_models[obs_var]
        noisy_y_vals = np.array([error_model(y_val, rng) for y_val in y_vals])
        res[obs_var] = {
            "t": np.copy(trajectory["t"]),
            "y": noisy_y_vals,
        }
    return res


def proportional_normal_error_model(
    noise_level: float, lower_bound: float | None = None
) -> ErrorModel:
    """
    Returns a noise model returning a noisy observation sampled from a
    normal random variable with mean equal to the ground truth observation
    value `obs` and standard deviation proportional to `obs * noise_level`
    where `noise_level` is expected to be nonnegative. If `lower_bound` is set,
    then the noise model returns the maximum of `lower_bound` and the
    sampled value described above.
    """
    if noise_level < 0:
        raise DomainError("Expected nonnegative noise_level")

    if lower_bound is None:
        lower_bound = -float("inf")

    def noise_func(obs: float, rng: Generator) -> float:
        eps = 1e-5
        noisy_obs = TruncNorm(mean=obs, mlx_omega=max(obs * noise_level, eps)).sample(
            rng
        )
        return max(noisy_obs, lower_bound)

    return noise_func
