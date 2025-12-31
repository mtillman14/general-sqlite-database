"""Tests for scidb.exceptions module."""

import pytest

from scidb.exceptions import (
    SciDBError,
    NotRegisteredError,
    NotFoundError,
    DatabaseNotConfiguredError,
    ReservedMetadataKeyError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from SciDBError."""

    def test_not_registered_error_is_scidb_error(self):
        assert issubclass(NotRegisteredError, SciDBError)

    def test_not_found_error_is_scidb_error(self):
        assert issubclass(NotFoundError, SciDBError)

    def test_database_not_configured_error_is_scidb_error(self):
        assert issubclass(DatabaseNotConfiguredError, SciDBError)

    def test_reserved_metadata_key_error_is_scidb_error(self):
        assert issubclass(ReservedMetadataKeyError, SciDBError)

    def test_scidb_error_is_exception(self):
        assert issubclass(SciDBError, Exception)


class TestExceptionMessages:
    """Test that exceptions can be raised with messages."""

    def test_scidb_error_with_message(self):
        with pytest.raises(SciDBError, match="test message"):
            raise SciDBError("test message")

    def test_not_registered_error_with_message(self):
        with pytest.raises(NotRegisteredError, match="MyClass not registered"):
            raise NotRegisteredError("MyClass not registered")

    def test_not_found_error_with_message(self):
        with pytest.raises(NotFoundError, match="No data found"):
            raise NotFoundError("No data found")

    def test_database_not_configured_error_with_message(self):
        with pytest.raises(DatabaseNotConfiguredError, match="not configured"):
            raise DatabaseNotConfiguredError("Database not configured")

    def test_reserved_metadata_key_error_with_message(self):
        with pytest.raises(ReservedMetadataKeyError, match="reserved key"):
            raise ReservedMetadataKeyError("Cannot use reserved key")


class TestExceptionCatching:
    """Test that exceptions can be caught properly."""

    def test_catch_specific_exception(self):
        try:
            raise NotFoundError("test")
        except NotFoundError as e:
            assert str(e) == "test"

    def test_catch_base_exception(self):
        """All scidb exceptions should be catchable via SciDBError."""
        exceptions_to_test = [
            NotRegisteredError("test"),
            NotFoundError("test"),
            DatabaseNotConfiguredError("test"),
            ReservedMetadataKeyError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except SciDBError:
                pass  # Should be caught
            else:
                pytest.fail(f"{type(exc).__name__} was not caught by SciDBError")
