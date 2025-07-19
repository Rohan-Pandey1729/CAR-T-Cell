from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pident.common.models import ODEModel


def _owens_bozic_ode_func(t: float, y: Any, params: Any) -> ArrayLike:
    T, E, C, M = y
    a, b_inv, dE, dC, g, jE, jC, K, k, l, mE, mC, qE, qC, s, KT, KE, KC, gamma = params

    b = 1 / b_inv

    C = max(C, 0)
    E = max(E, 0)

    tol = 1e-10

    if T < tol:
        DE = 0
    elif E < T:
        DE = dE * (E / T) ** l / (s + (E / T) ** l) * T
    else:
        DE = dE * (1 - s / (s + (T / E) ** (-l))) * T

    if T < tol:
        DC = 0
    elif C < T:
        DC = dC * (C / T) ** l / (s + (C / T) ** l) * T
    else:
        DC = dC * (1 - s / (s + (T / C) ** (-l))) * T

    dT_dt = a * T * (1 - b * T) - DE - DC - KT * (1 - np.exp(-M)) * T
    dE_dt = (
        g
        - mE * E
        - jE * np.log((E + C) / K) * (DE**2) / (k + DE**2) * E
        - qE * E * T
        - KE * (1 - np.exp(-M)) * E
    )
    dC_dt = (
        -mC * C
        - jC * np.log((E + C) / K) * (DC**2) / (k + DC**2) * C
        - qC * C * T
        - KC * (1 - np.exp(-M)) * C
    )
    dM_dt = -gamma * M

    return [dT_dt, dE_dt, dC_dt, dM_dt]


owens_bozic_param_names = [
    "a",
    "b_inv",
    "dE",
    "dC",
    "g",
    "jE",
    "jC",
    "K",
    "k",
    "l",
    "mE",
    "mC",
    "qE",
    "qC",
    "s",
    "KT",
    "KE",
    "KC",
    "gamma",
]

owens_bozic_obs_var_names = ["T", "E", "C", "M"]

owens_bozic_model = ODEModel(
    _owens_bozic_ode_func,
    owens_bozic_param_names.copy(),
    owens_bozic_obs_var_names.copy(),
)
