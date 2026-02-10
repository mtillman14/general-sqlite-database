"""Batch execution utilities for running functions over metadata combinations."""

from itertools import product
from typing import Any, Callable

from .fixed import Fixed


def for_each(
    fn: Callable,
    inputs: dict[str, type | Fixed],
    outputs: list[type],
    dry_run: bool = False,
    save: bool = True,
    pass_metadata: bool | None = None,
    **metadata_iterables: list[Any],
) -> None:
    """
    Execute a function for all combinations of metadata, loading inputs
    and saving outputs automatically.

    This function is loosely coupled to the storage layer. It expects:
    - Input types to have a `.load(**metadata)` class method
    - Output types to have a `.save(data, **metadata)` class method

    Args:
        fn: The function to execute (typically a thunked function, but
            works with any callable)
        inputs: Dict mapping parameter names to variable types
                (or Fixed wrappers for overridden metadata)
        outputs: List of variable types for outputs (positional)
        dry_run: If True, only print what would be loaded/saved without
                 actually executing
        save: If True (default), save each function run's output.
              If False, do not save any outputs.
        pass_metadata: If True, pass metadata values as keyword arguments
                      to the function. If None (default), auto-detect based
                      on fn's generates_file attribute.
        **metadata_iterables: Iterables of metadata values to combine

    Example:
        for_each(
            filter_data,
            inputs={"step_length": StepLength, "step_width": StepWidth},
            outputs=[FilteredStepLength, FilteredStepWidth],
            subject=subjects,
            session=sessions,
            speed=speeds,
        )

        # Preview what would happen
        for_each(
            filter_data,
            inputs={...},
            outputs=[...],
            dry_run=True,
            subject=subjects,
            ...
        )
    """
    keys = list(metadata_iterables.keys())
    value_lists = [metadata_iterables[k] for k in keys]

    # Count total iterations for progress
    total = 1
    for v in value_lists:
        total *= len(v)

    fn_name = getattr(fn, "__name__", repr(fn))
    should_pass_metadata = pass_metadata if pass_metadata is not None else getattr(fn, 'generates_file', False)

    if dry_run:
        print(f"[dry-run] for_each({fn_name})")
        print(f"[dry-run] {total} iterations over: {keys}")
        print(f"[dry-run] inputs: {_format_inputs(inputs)}")
        print(f"[dry-run] outputs: {[o.__name__ for o in outputs]}")
        print()

    completed = 0
    skipped = 0

    for values in product(*value_lists):
        metadata = dict(zip(keys, values))
        metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())

        if dry_run:
            _print_dry_run_iteration(inputs, outputs, metadata, should_pass_metadata)
            completed += 1
            continue

        # Load inputs
        loaded_inputs = {}
        load_failed = False

        for param_name, var_spec in inputs.items():
            if isinstance(var_spec, Fixed):
                load_metadata = {**metadata, **var_spec.fixed_metadata}
                var_type = var_spec.var_type
            else:
                load_metadata = metadata
                var_type = var_spec

            try:
                loaded_inputs[param_name] = var_type.load(**load_metadata)
            except Exception as e:
                print(f"[skip] {metadata_str}: failed to load {param_name} ({var_type.__name__}): {e}")
                load_failed = True
                break

        if load_failed:
            skipped += 1
            continue

        # Call the function
        print(f"[run] {metadata_str}: {fn_name}({', '.join(loaded_inputs.keys())})")

        # For plain functions (not Thunks), unwrap BaseVariable / ThunkOutput
        # inputs to raw .data so existing functions work without modification.
        # Thunks handle their own unwrapping internally.
        if not _is_thunk(fn):
            loaded_inputs = {
                k: _unwrap(v) for k, v in loaded_inputs.items()
            }

        try:
            if should_pass_metadata:
                result = fn(**loaded_inputs, **metadata)
            else:
                result = fn(**loaded_inputs)
        except Exception as e:
            print(f"[skip] {metadata_str}: {fn_name} raised: {e}")
            skipped += 1
            continue

        # Normalize single output to tuple
        if not isinstance(result, tuple):
            result = (result,)

        # Save outputs
        if save:
            for output_type, output_value in zip(outputs, result):
                try:
                    output_type.save(output_value, **metadata)
                    print(f"[save] {metadata_str}: {output_type.__name__}")
                except Exception as e:
                    print(f"[error] {metadata_str}: failed to save {output_type.__name__}: {e}")

        completed += 1

    # Summary
    print()
    if dry_run:
        print(f"[dry-run] would process {total} iterations")
    else:
        print(f"[done] completed={completed}, skipped={skipped}, total={total}")


def _is_thunk(fn: Any) -> bool:
    """Check if fn is a thunk-lib Thunk (without hard dependency)."""
    try:
        from thunk.core import Thunk
        return isinstance(fn, Thunk)
    except ImportError:
        return False


def _unwrap(value: Any) -> Any:
    """Extract raw data from a loaded variable, pass everything else through."""
    if hasattr(value, 'data'):
        return value.data
    return value


def _format_inputs(inputs: dict[str, type | Fixed]) -> str:
    """Format inputs dict for display."""
    parts = []
    for name, var_spec in inputs.items():
        if isinstance(var_spec, Fixed):
            fixed_str = ", ".join(f"{k}={v}" for k, v in var_spec.fixed_metadata.items())
            parts.append(f"{name}: Fixed({var_spec.var_type.__name__}, {fixed_str})")
        else:
            parts.append(f"{name}: {var_spec.__name__}")
    return "{" + ", ".join(parts) + "}"


def _print_dry_run_iteration(
    inputs: dict[str, type | Fixed],
    outputs: list[type],
    metadata: dict[str, Any],
    pass_metadata: bool = False,
) -> None:
    """Print what would happen for one iteration in dry-run mode."""
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    print(f"[dry-run] {metadata_str}:")

    for param_name, var_spec in inputs.items():
        if isinstance(var_spec, Fixed):
            load_metadata = {**metadata, **var_spec.fixed_metadata}
            var_type = var_spec.var_type
        else:
            load_metadata = metadata
            var_type = var_spec

        load_str = ", ".join(f"{k}={v}" for k, v in load_metadata.items())
        print(f"  load {param_name} = {var_type.__name__}.load({load_str})")

    if pass_metadata:
        print(f"  pass metadata: {metadata_str}")

    for output_type in outputs:
        print(f"  save {output_type.__name__}.save(..., {metadata_str})")
