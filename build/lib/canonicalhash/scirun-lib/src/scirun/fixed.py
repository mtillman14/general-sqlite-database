"""Fixed metadata wrapper for for_each inputs."""

from typing import Any


class Fixed:
    """
    Wrapper to specify fixed metadata overrides for an input.

    Use this when an input should be loaded with different metadata
    than the current iteration's metadata.

    Example:
        # Always load baseline from session="BL", regardless of current session
        for_each(
            compare_to_baseline,
            inputs={
                "baseline": Fixed(StepLength, session="BL"),
                "current": StepLength,
            },
            outputs=[Delta],
            subject=subjects,
            session=sessions,
        )
    """

    def __init__(self, var_type: type, **fixed_metadata: Any):
        """
        Args:
            var_type: The variable type to load (must have a .load() method)
            **fixed_metadata: Metadata values that override the iteration metadata
        """
        self.var_type = var_type
        self.fixed_metadata = fixed_metadata
