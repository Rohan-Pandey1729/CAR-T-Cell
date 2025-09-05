"""
Defines an interface for sampling from multivariate
distributions and some implementations.
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
