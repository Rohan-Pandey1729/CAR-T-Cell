"""
Multivariate distribution sampling strategies for parameter generation.

Defines the interface (MultivarSampler type) and concrete implementations for
generating parameter samples from multivariate distributions. Includes a sampler
for independent sampling.

Used in synthetic data generation to create parameter samples for virtual patients
according to various dependency structures between parameters.
"""

from typing import Callable

from numpy.random import Generator
from numpy.typing import ArrayLike

from pident.common.distributions import UnivarDist

MultivarSampler = Callable[[Generator], ArrayLike]


def independent_sampler(univar_dists: list[UnivarDist]) -> MultivarSampler:
    """
    Returns a multivariate sampler that independently samples from
    the given list of univariate distributions.
    """

    def multivar_sampler(rng: Generator) -> ArrayLike:
        return [ud.sample(rng) for ud in univar_dists]

    return multivar_sampler
