"""Custom exceptions for scidb."""


class SciStackError(Exception):
    """Base exception for all scidb errors."""

    pass


class NotRegisteredError(SciStackError):
    """Raised when trying to save/load an unregistered variable type."""

    pass


class NotFoundError(SciStackError):
    """Raised when no matching data is found for the given metadata."""

    pass


class DatabaseNotConfiguredError(SciStackError):
    """Raised when trying to use implicit database before configuration."""

    pass


class ReservedMetadataKeyError(SciStackError):
    """Raised when user tries to use a reserved metadata key."""

    pass


class AmbiguousVersionError(SciStackError):
    """Raised when load() matches multiple variants and no branch filter narrows to one."""

    pass


class AmbiguousParamError(SciStackError):
    """Raised when a bare param name matches multiple namespaced keys in branch_params."""

    pass


