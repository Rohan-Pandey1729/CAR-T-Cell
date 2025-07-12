"""General ways of adding noise to ground truth observations"""

import random
from distributions import normal, lognormal


def proportional_normal_noise(noise_level: float, lower_bound: float | None = None):
    """
    the noisy observation will be normally distributed with ground truth as mean
    and |(ground truth) * noise_level| as standard deviation and will be
    set to `lower_bound` if lower
    """

    if noise_level < 0:
        raise ValueError("expected nonnegative noise value")
    
    if lower_bound is None:
        lower_bound = -float("inf")

    def noise_func(obs: float, seed: int) -> float:
        random.seed(seed)
        noisy_obs = normal(mu=obs, sigma=abs(noise_level * obs))()
        return max(noisy_obs, lower_bound)

    return noise_func


def constant_lognormal_noise(sigma: float, lower_bound: float | None = None):
    """
    sigma is standard deviation of the log of the noisy observation
    (which is a normally distributed random variable)
    """

    if lower_bound is None:
        lower_bound = -float("inf")

    def noise_func(obs: float, seed: int) -> float:
        if obs <= 0:
            raise ValueError(
                "observations can't be <= 0 if using lognormal distribution"
            )
        random.seed(seed)
        noisy_obs = lognormal(obs, sigma)()
        return max(noisy_obs, lower_bound)

    return noise_func
