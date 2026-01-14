"""
Provides ways to interact with probability distributions.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike
from scipy.stats import lognorm, truncnorm

from pident.common.exceptions import DomainError


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

    @abstractmethod
    def plot_bounds(self) -> tuple[float, float]:
        """
        Returns a recommended interval to plot the pdf on.
        """
        pass

    @property
    @abstractmethod
    def median(self) -> float:
        """
        Returns the median of this distribution.
        """
        pass

    @property
    @abstractmethod
    def mlx_omega(self) -> float:
        """
        Returns the Monolix omega parameter for this distribution.
        """


class Constant(UnivarDist):
    def __init__(self, val: float):
        """
        Constructs a constant random variable always equal to `val`.
        """
        self._val = val
        self._tol = val * 0.01

    def sample(self, rng: Generator) -> float:
        return self._val

    def pdf(self, x: ArrayLike) -> ArrayLike:
        # have to do something that will show up on a plot
        x = np.array(x)
        return np.where(
            np.abs(x - self._val) < self._tol,
            np.ones(shape=x.shape, dtype=np.float64),
            np.zeros(shape=x.shape, dtype=np.float64),
        )

    def plot_bounds(self) -> tuple[float, float]:
        # assumes value is positive
        return (0.0, 2 * self._val)

    @property
    def median(self) -> float:
        return self._val

    @property
    def mlx_omega(self) -> float:
        return 0.0


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
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._mean = mean
        self._mlx_omega = mlx_omega

    def sample(self, rng: Generator) -> float:
        return self._rv.rvs(random_state=rng)

    def pdf(self, x: ArrayLike) -> ArrayLike:
        return self._rv.pdf(x)  # type: ignore

    def plot_bounds(self) -> tuple[float, float]:
        MIN_QUANTILE = 0.01
        MAX_QUANTILE = 0.99
        WIDTH_FACTOR = 0.2
        no_trunc_low, no_trunc_high = (
            self._rv.ppf(MIN_QUANTILE),
            self._rv.ppf(MAX_QUANTILE),
        )
        no_trunc_width = no_trunc_high - no_trunc_low
        low = no_trunc_low
        if self._lower_bound > -np.inf:
            low = self._lower_bound - WIDTH_FACTOR * no_trunc_width
        high = no_trunc_high
        if self._upper_bound < np.inf:
            high = self._upper_bound + WIDTH_FACTOR * no_trunc_width
        return (low, high)

    @property
    def median(self) -> float:
        return self._rv.ppf(0.5)

    @property
    def mlx_omega(self) -> float:
        return self._mlx_omega


class TruncLognorm(UnivarDist):
    _MAX_ITER = 100

    def __init__(
        self, median: float, mlx_omega: float, lower_bound=0.0, upper_bound=np.inf
    ):
        """
        Constructs a lognormal random variable `Y` with median `median`
        such that `X = log(Y)` is normal with
        standard deviation `mlx_omega`; then truncates `Y`
        below at `lower_bound`
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
        scale = median
        self._rv = lognorm(s=s, scale=scale)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._norm_factor = 1.0 / (
            self._rv.cdf(upper_bound) - self._rv.cdf(lower_bound)
        )  # pdf normalization factor
        self._median = median
        self._mlx_omega = mlx_omega

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
            (self._lower_bound <= x) & (x <= self._upper_bound),
            self._rv.pdf(x) * self._norm_factor,  # type: ignore
            0.0,
        )

    def plot_bounds(self) -> tuple[float, float]:
        MIN_QUANTILE = 0.0
        MAX_QUANTILE = 0.99
        WIDTH_FACTOR = 0.25
        MAX_DISTORTION_FACTOR = 6.0
        no_trunc_low, no_trunc_high = (
            self._rv.ppf(MIN_QUANTILE),
            self._rv.ppf(MAX_QUANTILE),
        )
        no_trunc_width = no_trunc_high - no_trunc_low
        low = no_trunc_low
        if self._lower_bound > 0:
            low = max(0.0, self._lower_bound - WIDTH_FACTOR * no_trunc_width)
        high = no_trunc_high
        if self._upper_bound < np.inf:
            high = min(high, self._upper_bound + WIDTH_FACTOR * no_trunc_width)
        return (low, min(high, MAX_DISTORTION_FACTOR * self._median))

    @property
    def median(self) -> float:
        return self._median

    @property
    def mlx_omega(self) -> float:
        return self._mlx_omega
