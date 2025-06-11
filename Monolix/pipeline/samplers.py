"""General ways of sampling from a distribution"""

import random
from typing import Callable


def normal_sampler(param_distributions: list[tuple[float, float]]):
    """
    Creates a sampler function for normally distributed parameters

    Args:
        param_distributions: List of (mean, std) pairs for each parameter

    Returns:
        Function that samples parameters given n_indivs and seed
    """

    def sampler(n_indivs: int, seed: int) -> list[list[float]]:
        """
        Sample parameters for n individuals

        Args:
            n_indivs: Number of individuals to sample for
            seed: Random seed for reproducibility

        Returns:
            List of parameter lists, one for each individual
        """
        random.seed(seed)

        sampled_params = []
        for _ in range(n_indivs):
            individual_params = []
            for mean, std in param_distributions:
                param_value = random.gauss(mean, std)
                individual_params.append(param_value)
            sampled_params.append(individual_params)

        return sampled_params

    return sampler


def independent_sampler(one_var_samplers: list[Callable[[], float]]):
    """
    Accepts a list of 1-variable samplers, which should use `random` for randomness,
    each of which should take no arguments and return a random parameter value.
    Returns a sampler function which samples `n_indiv` parameter sets, where
    within each set the values are sampled independently.
    """

    def sampler(n_indivs: int, seed: int) -> list[list[float]]:
        random.seed(seed)
        sampled_params = []
        for _ in range(n_indivs):
            sampled_params.append([s() for s in one_var_samplers])
        return sampled_params

    return sampler
