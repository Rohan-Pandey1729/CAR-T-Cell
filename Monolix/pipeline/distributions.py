import random
from math import exp, log
from typing import Callable


def truncate(
    distribution_sampler: Callable[[], float],
    low: float | None = None,
    high: float | None = None,
):
    """
    Returns a new function which repeatedly samples values using
    `distribution_sampler` until the given value is at least `low`
    (if given) and at most `high` (if given)
    """
    if low and high and low > high:
        raise ValueError("low must be <= high")

    if not low:
        low = -float("inf")

    if not high:
        high = float("inf")

    def truncated_sampler():
        val = distribution_sampler()
        while not (low <= val <= high):
            val = distribution_sampler()
        return val

    return truncated_sampler


def normal(mu: float, sigma: float):
    """
    Returns a function that samples from a normal distribution
    with mean `mu` and standard deviation `sigma`
    """
    return lambda: random.gauss(mu, sigma)


def lognormal(mu: float, sigma: float):
    """
    Returns a function that samples from a lognormal distribution
    with mean `mu` and where the log of the associated random variable
    has standard deviation `sigma`
    """
    if mu <= 0:
        raise ValueError("mean can't be <= 0 if using lognormal distribution")
    noise = random.gauss(sigma=sigma)
    return lambda: exp(log(mu) + noise)

def constant(val: float):
    """
    Returns a function that always returns `val`
    """
    return lambda: val
