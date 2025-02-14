"""
Utility functions for generating mock data
based on the nonlinear mixed-effects model.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.integrate import odeint

type RealToReal = Callable[[float], float]

# (y, t, params) => derivatives
type ODEModel = Callable[[float, float, list[float]], list[float]]
type ErrorModel = list[Literal["combined1"] | Literal["combined2"]]


@dataclass
class Distribution:
    """
    Contains properties `forward` and `inv`
    such that `forward(X)` is to be normally distributed
    for `X` drawn from this distribution and `inv` is the inverse
    of `forward`.

    In https://monolixsuite.slp-software.com/monolix/2024R1/individual-model,
    `forward` takes the role of `h`.
    """

    forward: RealToReal
    inv: RealToReal


distributions = {
    "normal": Distribution(forward=lambda x: x, inv=lambda x: x),
    "lognormal": Distribution(forward=math.log, inv=math.exp),
}


# we'll add a wrapper function around everything eventually


def sample_parameters(
    pop_params: list[float],
    distributions: list[Distribution],
    pop_stds: list[float],
    n_samples: int,
    seed=42,
) -> list[list[float]]:
    """
    Generate n samples of parameters based on population parameters,
    a distribution type to sample the parameters from,
    and standard deviations of the parameters after transforming
    them to be distributed normally (e.g. if sampling from lognormal,
    distribution, then this would be the standard deviation
    after taking the log of the parameter). Does not factor in covariates;
    this should be handled elsewhere if applicable.

    Args:
    - pop_params: List of mean values for the parameters.
    - distributions: Distribution type to sample the parameters from.
    - pop_stds: List of standard deviations for the transformed parameters.
    - n_samples: Number of samples to generate.

    Returns:
    - List of n sampled parameter lists, with parameters in the same order
        as in `pop_params` and `pop_stds`.
    """
    assert len(pop_params) == len(pop_stds) == len(distributions)
    random.seed(seed)
    return [
        [
            distribution.inv(random.gauss(distribution.forward(mean), std))
            for mean, std, distribution in zip(pop_params, pop_stds, distributions)
        ]
        for _ in range(n_samples)
    ]


def simulate_model(
    model: ODEModel,
    sampled_params_all: list[list[float]],
    initial_conditions_all: list[list[float]],
    obs_times_all: list[list[float]],
    error_models: ErrorModel,
    error_dists: list[Distribution],
    err_a: float,
    err_b=0.0,
    err_c=1.0,
    seed=42,
) -> list[np.ndarray]:
    """
    Simulate observations for multiple parameter sets based on
    a "structural" model, each associated
    with initial conditions, times to make observations at,
    and an error model and distribution defining how noise should be
    added to the structural model's predictions. The error model behavior
    and the magnitude of the prediction-observation errors
    are controlled by three additional parameters; see
    https://monolixsuite.slp-software.com/monolix/2024R1/observation-error-model.

    Args:
    - model: Function that returns a list of time derivatives for \
        each variable given the time and a set of parameters
    - sampled_params_all: List of parameter lists
    - initial_conditions_all: List of lists of initial conditions for each variable
    - obs_times_all: List of lists of times to record observations at
    - error_models: List of error models defining how noise is added to raw predictions
    - error_dists: List of distributions to sample errors from
    - err_a: Parameter controlling magnitude of "constant" error
    - err_b: Parameter controlling magnitude of "proportional" error depending on \
        observation value
    - err_c: Exponent of observation value t use in "proportional" errors

    Returns:
    - List of arrays of observations, where rows in each array represent \
        a timeseries of observations for a single variable
    """

    assert (
        0 < len(sampled_params_all) == len(initial_conditions_all) == len(obs_times_all)
    )

    assert len(initial_conditions_all[0]) == len(error_models) == len(error_dists)

    random.seed(seed)
    # each element of the list below is for an individual
    # for each element, the rows are a timeseries of observations for a single variable
    observations_all: list[np.ndarray] = [
        odeint(model, initial_conditions, times, args=(params,)).T
        for initial_conditions, times, params in zip(
            initial_conditions_all, obs_times_all, sampled_params_all
        )
    ]

    for observations in observations_all:
        for row, error_model, error_dist in zip(
            observations, error_models, error_dists
        ):
            for i, _ in enumerate(row):
                transformed_obs = error_dist.forward(row[i])
                error = 0.0
                match error_model:
                    case "combined1":
                        error = (
                            err_a + err_b * transformed_obs**err_c
                        ) * random.gauss()
                    case "combined2":
                        error = (
                            math.sqrt(err_a**2 + (err_b * transformed_obs**err_c) ** 2)
                            * random.gauss()
                        )
                    case _:
                        raise ValueError(f"Invalid error model {error_model}")
                row[i] = error_dist.inv(transformed_obs + error)
    return observations_all


def generate_sample_csv(
    save_name: str,
    model: ODEModel,
    pop_param_info: list[tuple[str, float, float, Distribution]],
    obs_var_info: list[
        tuple[str, float, float, Distribution, ErrorModel, Distribution]
    ],
    n_indivs: int,
    n_obs: int,
    max_day_number: int,
    err_a: float,
    err_b: float,
    err_c=1.0,
    obs_times_all: list[list[int]] = None,
    seed=42,
):
    """
    Generates and saves a CSV containing observations for multiple individuals
    in a format suitable for use by Monolix.

    Args:
    - save_name: Filename to save the CSV as (without .csv)
    - model: Function that returns a list of time derivatives for \
        each variable to observe given the time and a set of parameters
    - pop_param_info: List of tuples, each corresponding to a model parameter. \
        Each contains the following: \
        (name, population mean, population stdev, error distribution). \
        See https://monolixsuite.slp-software.com/monolix/2024R1/individual-model
    - obs_var_info: List of tuples, each corresponding to an observed variable. \
        Each contains the following: \
        (name, avg initial condition, initial condition stdev, \
        initial condition error distribution, \
        observation error model, observation error distribution). \
        See https://monolixsuite.slp-software.com/monolix/2024R1/observation-error-model
        for more details about the last two
    - n_indivs: Number of individuals to generate observations for. \
        Overridden if `obs_times_all` is passed.
    - n_obs: Approximate number of observations to generate per person. \
        Overridden if `obs_times_all` is passed.
    - max_day_number: Last day which might have an observation. \
        Overridden if `obs_times_all` is passed.
    - err_a: Parameter controlling magnitude of "constant" error
    - err_b: Parameter controlling magnitude of "proportional" error depending on \
        observation value
    - err_c: Exponent of observation value t use in "proportional" errors
    - obs_times_all: List of lists of days to collect observations on
    """
    param_names, pop_params, pop_stds, pop_dists = zip(*pop_param_info)
    (
        obs_var_names,
        init_conds,
        init_cond_stds,
        init_cond_error_dists,
        obs_error_models,
        obs_error_dists,
    ) = zip(*obs_var_info)

    params_and_init_conds = pop_params + init_conds
    param_and_init_cond_stds = pop_stds + init_cond_stds
    param_and_init_error_dists = pop_dists + init_cond_error_dists

    sampled_params_and_init_conds = sample_parameters(
        params_and_init_conds,
        param_and_init_error_dists,
        param_and_init_cond_stds,
        n_indivs,
        seed=seed,
    )
    sampled_params_all = [x[: len(pop_params)] for x in sampled_params_and_init_conds]
    initial_conditions_all = [
        x[len(pop_params) :] for x in sampled_params_and_init_conds
    ]

    if not obs_times_all:
        BASE_OBS_INTERVAL = -1 + max_day_number // n_obs
        obs_times_all = [
            [
                max(0, BASE_OBS_INTERVAL * j - (i * j % 7 + j) % 3)
                for j in range(n_obs - (i % 3))
            ]
            for i in range(n_indivs)
        ]

    observations_all = simulate_model(
        model,
        sampled_params_all,
        initial_conditions_all,
        obs_times_all,
        obs_error_models,
        obs_error_dists,
        err_a,
        err_b,
        err_c,
        seed=seed,
    )

    obs_dict = {
        "time": [],
        "id": [],
        "observation": [],
        "observation_id": [],
        "observation_type": [],
    }

    for obs_id, var_obs_all in enumerate(zip(*observations_all)):
        var_name = obs_var_names[obs_id]
        for indiv_id, (var_obs, obs_times) in enumerate(
            zip(var_obs_all, obs_times_all)
        ):
            for obs, time in zip(var_obs, obs_times):
                obs_dict["time"].append(time)
                obs_dict["id"].append(indiv_id)
                obs_dict["observation"].append(np.log(obs) / np.log(10))
                obs_dict["observation_id"].append(obs_id)
                obs_dict["observation_type"].append(var_name)

    pd.DataFrame(obs_dict).to_csv(
        save_name if save_name.endswith(".csv") else f"{save_name}.csv", index=False
    )

    return obs_times_all, observations_all, obs_dict
