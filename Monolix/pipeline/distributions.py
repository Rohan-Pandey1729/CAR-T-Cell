import random
from math import log, exp


def _clamp(val: float, low: float | None, high: float | None):
    if low and high and (low > high):
        raise ValueError("incompatible bounds")
    if high:
        val = min(val, high)
    if low:
        val = max(val, low)
    return val


def normal(
    mu: float, sigma: float, low: float | None = None, high: float | None = None
):
    return _clamp(random.gauss(mu, sigma), low, high)


def lognormal(
    mu: float, sigma: float, low: float | None = None, high: float | None = None
):
    if mu <= 0:
        raise ValueError("mean can't be <= 0 if using lognormal distribution")
    noise = random.gauss(sigma=sigma)
    return _clamp(exp(log(mu) + noise), low, high)
