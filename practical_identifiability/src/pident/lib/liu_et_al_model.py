from typing import Any

from numpy.typing import ArrayLike

from pident.common.models import ODEModel


def _liu_et_al_ode_func(t: float, y: Any, params: Any) -> ArrayLike:
    nP, nTA, nTN, nN = y
    rP, rTA, lTA, lTN, nC, e, KP, Kr, KA, kA, rN, km, kb, KN = params

    dnP_dt = rP * (1 - nP / nC) * nP - (e * nP / (nP + KP)) * nTA
    dnTA_dt = rTA * (nP / (nP + Kr)) * nTA + kA * (nP / (nP + KA)) * nTN - lTA * nTA
    dnTN_dt = -kA * (nP / (nP + KA)) * nTN - lTN * nTN
    dnN_dt = rN * (1 - nN / nC) * nN + km * nP - e / kb * nN / (nN + KN) * nTA

    return [dnP_dt, dnTA_dt, dnTN_dt, dnN_dt]


_liu_et_al_param_names = [
    "rP",
    "rTA",
    "lTA",
    "lTN",
    "nC",
    "e",
    "KP",
    "Kr",
    "KA",
    "kA",
    "rN",
    "km",
    "kb",
    "KN",
]

_liu_et_al_obs_var_names = ["nP", "nTA", "nTN", "nN"]

liu_et_al_model = ODEModel(
    _liu_et_al_ode_func,
    _liu_et_al_param_names,
    _liu_et_al_obs_var_names,
)


def _liu_et_al_no_nN_ode_func(t: float, y: Any, params: Any) -> ArrayLike:
    nP, nTA, nTN = y
    rP, rTA, lTA, lTN, nC, e, KP, Kr, KA, kA = params

    dnP_dt = rP * (1 - nP / nC) * nP - (e * nP / (nP + KP)) * nTA
    dnTA_dt = rTA * (nP / (nP + Kr)) * nTA + kA * (nP / (nP + KA)) * nTN - lTA * nTA
    dnTN_dt = -kA * (nP / (nP + KA)) * nTN - lTN * nTN

    return [dnP_dt, dnTA_dt, dnTN_dt]


_liu_et_al_no_nN_param_names = [
    "rP",
    "rTA",
    "lTA",
    "lTN",
    "nC",
    "e",
    "KP",
    "Kr",
    "KA",
    "kA",
]

_liu_et_al_no_nN_obs_var_names = ["nP", "nTA", "nTN"]

liu_et_al_no_nN_model = ODEModel(
    _liu_et_al_no_nN_ode_func,
    _liu_et_al_no_nN_param_names,
    _liu_et_al_no_nN_obs_var_names,
)
