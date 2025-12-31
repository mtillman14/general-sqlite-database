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
