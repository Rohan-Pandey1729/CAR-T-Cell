from collections.abc import Callable, Iterable

import numpy as np
from scipy.integrate import odeint

# for timeseries and parameter sets
type FloatArr = Iterable[float]


def generate_ground_truths(
    model: Callable,
    sampled_params: Iterable[FloatArr],
    initial_conditions: Iterable[FloatArr],
    observation_times: Iterable[FloatArr],
) -> list[np.ndarray]:
    """
    Note: the format of each ndarray is such that each **row** contains observations
    for each observation variables for a single timestep

    See [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)
    """
    return [
        odeint(model, ic, times, args=(params,))
        for ic, times, params in zip(
            initial_conditions, observation_times, sampled_params
        )
    ]
