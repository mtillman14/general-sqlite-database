# Define Processing Functions with @thunk
# Documentation: See docs/guide/lineage.md for the thunk system
#
# Key concept: @thunk wraps a function to:
# - Capture the function's bytecode hash (for versioning)
# - Track inputs when called (for lineage)
# - Return an OutputThunk that carries lineage information
#
# The n_outputs parameter tells thunk how many values the function returns.

import numpy as np

from thunk import thunk

@thunk(n_outputs=1)
def apply_moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a simple moving average filter.

    The @thunk decorator captures:
    - Function bytecode hash (changes if you edit this function)
    - Input arguments when called

    Documentation: docs/guide/lineage.md section "The @thunk Decorator"
    """
    kernel = np.ones(window_size) / window_size
    # Use 'same' mode to maintain array length
    return np.convolve(signal, kernel, mode='same')


@thunk(n_outputs=1)
def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [0, 1] range.

    This is a second processing step - chaining thunked functions
    automatically builds a computation graph for lineage tracking.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)


@thunk(n_outputs=1)
def compute_statistics(signal: np.ndarray) -> dict:
    """
    Compute summary statistics for a signal.

    Returns a dict which will be stored using SignalStatistics variable type.
    """
    return {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'min': float(np.min(signal)),
        'max': float(np.max(signal)),
        'n_samples': len(signal)
    }