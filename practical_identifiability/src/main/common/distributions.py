"""
Provides ways to interact with probability distributions.
"""

from abc import ABC, abstractmethod

import numpy as np
from main.common.exceptions import DomainError
from numpy.random import Generator
from numpy.typing import ArrayLike
from scipy.stats import lognorm, truncnorm


class SamplingError(RuntimeError):
    """
    An error occurred during sampling.
    """

    pass


class UnivarDist(ABC):
    @abstractmethod
    def sample(self, rng: Generator) -> float:
        """
        Samples a single value from this distribution using the
        provided `rng`.
        """
        pass

    @abstractmethod
    def pdf(self, x: ArrayLike) -> ArrayLike:
        """
        Returns the evaluation of this distribution's pdf at `x`
        (which may be a scalar or array).
        """
        pass


class TruncNorm(UnivarDist):
    def __init__(
        self, mean: float, mlx_omega: float, lower_bound=-np.inf, upper_bound=np.inf
    ):
        """
        Constructs a normal random variable with mean `mean` and
        standard deviation `mlx_omega`, truncated below at `lower_bound`
        and above at `upper_bound` (which should be greater than `lower_bound`).

        Default value `lower_bound = -np.inf` prevents truncation from below
        and default `upper_bound = np.inf` prevents truncation from above.
        """
        if mlx_omega <= 0:
            raise DomainError("mlx_omega must be positive")
        if upper_bound <= lower_bound:
            raise DomainError("upper_bound must be greater than lower_bound")

        self._rv = truncnorm(loc=mean, scale=mlx_omega, a=lower_bound, b=upper_bound)

    def sample(self, rng: Generator) -> float:
        return self._rv.rvs(random_state=rng)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return self._rv.pdf(x)  # type: ignore


class TruncLognorm(UnivarDist):
    _MAX_ITER = 100

    def __init__(
        self, mean: float, mlx_omega: float, lower_bound=0.0, upper_bound=np.inf
    ):
        """
        Constructs a lognormal random variable `Y` such that
        `X = log(Y)` is normal with mean `mean` and
        standard deviation `mlx_omega`, truncated below at `lower_bound`
        (which should be nonnegative) and above at `upper_bound`
        (which should be greater than `lower_bound`).

        Default value `lower_bound = 0` prevents truncation from below
        and default `upper_bound = np.inf` prevents truncation from above.
        """
        if mlx_omega <= 0:
            raise DomainError("mlx_omega must be positive")
        if lower_bound < 0:
            raise DomainError("lower_bound should be nonnegative")
        if upper_bound <= lower_bound:
            raise DomainError("upper_bound must be greater than lower_bound")

        s = mlx_omega
        scale = np.exp(mean)
        self._rv = lognorm(s=s, scale=scale)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._norm_factor = 1.0 / (
            self._rv.cdf(upper_bound) - self._rv.cdf(lower_bound)
        )  # pdf normalization factor

    def sample(self, rng: Generator) -> float:
        for _ in range(TruncLognorm._MAX_ITER):
            if (
                self._lower_bound
                <= (val := self._rv.rvs(random_state=rng))
                <= self._upper_bound
            ):
                return val
        raise SamplingError(
            f"Failed to sample value after {TruncLognorm._MAX_ITER} attempts"
        )

    def pdf(self, x: ArrayLike) -> ArrayLike:
        x = np.array(x)
        return np.where(
            self._lower_bound <= x <= self._upper_bound,
            self._rv.pdf(x) * self._norm_factor,  # type: ignore
            0.0,
        )
