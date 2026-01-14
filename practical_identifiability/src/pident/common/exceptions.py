"""
Shared exceptions for the pident package.

Provides custom exception types for parameter validation and domain checking
across the package.
"""


class DomainError(ValueError):
    """
    Raised when a parameter value is not in its expected domain.

    Used for validation of parameter values (e.g., negative values where
    positive values are required) and distribution parameters.
    """

    pass
