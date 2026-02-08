"""Exceptions for the scidb network layer."""


class SciDBNetError(Exception):
    """Base exception for scidb-net errors."""
    pass


class NetworkError(SciDBNetError):
    """Raised when an HTTP request fails (connection error, timeout, etc.)."""
    pass


class ServerError(SciDBNetError):
    """Raised when the server returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class SerializationError(SciDBNetError):
    """Raised when data serialization or deserialization fails."""
    pass
