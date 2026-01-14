from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

# f(t, y, params) -> dy_dt
ODEFunc = Callable[[float, ArrayLike, ArrayLike], ArrayLike]
GroundTruthFunc = Callable[[ArrayLike], dict[str, ArrayLike]]
IntegrationMethod = Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]


class DuplicateNameError(ValueError):
    pass


class OutOfBoundsError(ValueError):
    pass


class ODEModel:
    """
    This is a wrapper around `scipy.integrate.solve_ivp` for a fixed
    ODE model which associates model parameters with names.
    """

    def __init__(
        self, ode_func: ODEFunc, param_names: list[str], obs_var_names: list[str]
    ):
        """
        Constructs an instance for an ODE model with the given system of
        differential equations.

        `ode_func` should be a function of the form `f(t, y, params) -> dy_dt`
        as expected by `scipy.integrate.solve_ivp`, where `y` is an ndarray of
        observation variables, `params` is an ndarray of model parameters,
        and the return value `dy_dt` is an ndarray of time derivatives for
        each observation variable.

        `param_names` should be a list of variable names for the parameters `ode_func`
        expects in the same order, and should not contain duplicates.
        Behavior is undefined if this is the wrong length.

        `obs_var_names` should be a list of variable names for the observation variables
        `ode_func` works with in the same order, and should not contain duplicates.
        Behavior is undefined if this is the wrong length.
        """
        if len(param_names) != len(set(param_names)):
            raise DuplicateNameError("Duplicate param name")
        if len(obs_var_names) != len(set(obs_var_names)):
            raise DuplicateNameError("Duplicate observation variable name")
        self._ode_func = ode_func
        self._param_names = param_names
        self._obs_var_names = obs_var_names

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @property
    def obs_var_names(self) -> list[str]:
        return self._obs_var_names

    def get_ground_truth(
        self,
        param_values: dict[str, float] | ArrayLike,
        initial_values: dict[str, float] | ArrayLike,
        t_min: float,
        t_max: float,
        int_method: IntegrationMethod = "LSODA",
    ) -> GroundTruthFunc:
        """
        Returns the ground truth trajectories between times `t_min` and `t_max`
        for the given parameter values and initial observation variable values.

        `param_values` can either be an ndarray of parameter values for the
        ODE model, in the same order as the differential equation implementation
        expects, or a dict mapping parameter variable names to their values,
        where the names used must be the same as the ones used to initialize
        this object.

        `initial_values` can either be an ndarray of observation variable
        initial values for the ODE model, in the same order as
        the differential equation implementation expects,
        or a dict mapping observation variable names to initial values values,
        where the names used must be the same as the ones used to initialize
        this object.

        The return value is a function accepting a single parameter
        `t`, which may be a scalar or 1d ndarray, and returning the following:

        - if `t` is a scalar, then a 1d ndarray is returned containing the
          value of each observation variable at `t`.
        - if `t` is a 1d ndarray, then a 2d ndarray is returned where each row
          contains the values of a single observation variable at times `t`.

        In both cases, the observation variables that elements of the return value
        correspond to are in the same order as outputted from the differential equation
        implementation for this ODE model.
        """
        if isinstance(param_values, dict):
            param_values = np.array([param_values[name] for name in self._param_names])
        else:
            param_values = np.array(param_values)
        if isinstance(initial_values, dict):
            initial_values = np.array(
                [initial_values[name] for name in self._obs_var_names]
            )
        else:
            initial_values = np.array(initial_values)
        sol = solve_ivp(
            self._ode_func,
            (t_min, t_max),
            initial_values,
            args=[param_values],
            dense_output=True,
            method=int_method,
        )

        def ground_truth(t: ArrayLike) -> dict[str, ArrayLike]:
            t = np.array(t)
            if not np.all((t_min <= t) & (t <= t_max)):
                raise OutOfBoundsError(f"Times must be between {t_min} and {t_max}")

            gt_vals = sol.sol(t)
            return {
                obs_var_name: gt_val
                for obs_var_name, gt_val in zip(self._obs_var_names, gt_vals)
            }

        return ground_truth
