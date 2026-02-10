"""Custom exceptions for scidb."""


class SciDBError(Exception):
    """Base exception for all scidb errors."""

    pass


class NotRegisteredError(SciDBError):
    """Raised when trying to save/load an unregistered variable type."""

    pass


class NotFoundError(SciDBError):
    """Raised when no matching data is found for the given metadata."""

    pass


class DatabaseNotConfiguredError(SciDBError):
    """Raised when trying to use implicit database before configuration."""

    pass


class ReservedMetadataKeyError(SciDBError):
    """Raised when user tries to use a reserved metadata key."""

    pass


class UnsavedIntermediateError(SciDBError):
    """Raised when strict lineage mode detects an unsaved intermediate variable.

    In strict mode, all upstream BaseVariables must be saved before saving
    downstream results. This ensures complete data provenance and enables
    cache hits at every step.
    """

    pass
