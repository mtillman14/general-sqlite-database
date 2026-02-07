"""Path generation from templates and metadata combinations."""

from itertools import product
from pathlib import Path
from typing import Any, Iterator


class PathGenerator:
    """
    Generate file paths with associated metadata from a template and metadata values.

    This class helps separate path resolution from data loading by generating
    all combinations of metadata values along with their fully resolved paths.

    Args:
        path_template: A format string with placeholders for metadata fields.
                      Uses Python's str.format() syntax, e.g., "{subject}/trial_{trial}.mat"
        root_folder: Optional root folder path. If provided, paths are resolved
                    relative to this folder. If None, paths are resolved
                    relative to the current working directory (via Path.resolve()).
        **metadata: Keyword arguments where each key is a metadata field name and
                   each value is an iterable of values for that field.

    Example:
        >>> paths = PathGenerator(
        ...     "{subject}/trial_{trial}.mat",
        ...     root_folder="/data/experiment",
        ...     subject=range(3),
        ...     trial=range(5)
        ... )

        >>> for path, meta in paths:
        ...     print(path, meta)
        # /data/experiment/0/trial_0.mat {'subject': 0, 'trial': 0}
        # /data/experiment/0/trial_1.mat {'subject': 0, 'trial': 1}
        # ...

        >>> # Can also access by index or get length
        >>> print(len(paths))  # 15
        >>> path, meta = paths[0]
        >>> print(path)  # /data/experiment/0/trial_0.mat
        >>> print(meta)  # {'subject': 0, 'trial': 0}
    """

    def __init__(
        self,
        path_template: str,
        root_folder: str | Path | None = None,
        **metadata: Any,
    ):
        self.path_template = path_template
        self.root_folder = Path(root_folder) if root_folder is not None else None
        self.metadata_keys = list(metadata.keys())
        self.metadata_values = [list(v) for v in metadata.values()]

        # Pre-compute all combinations for efficient indexing
        self._items: list[tuple[Path, dict[str, Any]]] = []
        for combo in product(*self.metadata_values):
            meta = dict(zip(self.metadata_keys, combo))
            relative_path = Path(path_template.format(**meta))
            if self.root_folder is not None:
                full_path = (self.root_folder / relative_path).resolve()
            else:
                full_path = relative_path.resolve()
            self._items.append((full_path, meta))

    def __iter__(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        """Iterate over all path/metadata combinations."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return the total number of path/metadata combinations."""
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Path, dict[str, Any]]:
        """Get a specific path/metadata combination by index."""
        return self._items[index]

    def __repr__(self) -> str:
        """Return a string representation of the PathGenerator."""
        return (
            f"PathGenerator({self.path_template!r}, "
            f"root_folder={self.root_folder!r}, "
            f"{', '.join(f'{k}=...' for k in self.metadata_keys)})"
        )

    def to_list(self) -> list[tuple[Path, dict[str, Any]]]:
        """Return all path/metadata combinations as a list."""
        return list(self._items)
