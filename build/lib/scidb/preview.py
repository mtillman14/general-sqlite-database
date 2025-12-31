"""Preview generation for human-readable data summaries."""

import numpy as np
import pandas as pd


def generate_preview(df: pd.DataFrame, max_length: int = 500) -> str:
    """
    Generate a human-readable preview string for a DataFrame.

    The preview is designed to be viewable in SQLite browsers and provides
    a quick summary of the data without needing Python.

    Args:
        df: The DataFrame to preview
        max_length: Maximum length of the preview string

    Returns:
        A human-readable string summarizing the data
    """
    if df.empty:
        return "(empty)"

    rows, cols = df.shape
    parts = []

    # Header with shape
    parts.append(f"[{rows} rows x {cols} cols]")

    # Column info
    col_names = list(df.columns)
    if len(col_names) <= 5:
        parts.append(f"Columns: {', '.join(str(c) for c in col_names)}")
    else:
        shown = [str(c) for c in col_names[:3]] + ["..."] + [str(col_names[-1])]
        parts.append(f"Columns: {', '.join(shown)}")

    # For single-row DataFrames (scalars, dicts), show the values directly
    if rows == 1:
        values = []
        for col in df.columns:
            val = df[col].iloc[0]
            val_str = _format_value(val)
            values.append(f"{col}={val_str}")
        values_str = ", ".join(values)
        if len(values_str) > max_length - 50:
            values_str = values_str[:max_length - 53] + "..."
        parts.append(f"Values: {values_str}")
        return " | ".join(parts)

    # For multi-row DataFrames, show stats and sample values
    # Try to identify the "value" column (common patterns)
    value_col = None
    for candidate in ["value", "values", "data", "val"]:
        if candidate in df.columns:
            value_col = candidate
            break

    # If no standard value column, use the last numeric column
    if value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[-1]

    if value_col is not None:
        col_data = df[value_col]
        dtype = col_data.dtype

        # Numeric stats
        if np.issubdtype(dtype, np.number):
            try:
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                parts.append(
                    f"{value_col}: min={_format_value(min_val)}, "
                    f"max={_format_value(max_val)}, "
                    f"mean={_format_value(mean_val)}"
                )
            except (TypeError, ValueError):
                pass

        # Show first few values
        sample_size = min(5, len(col_data))
        sample = [_format_value(v) for v in col_data.iloc[:sample_size]]
        sample_str = ", ".join(sample)
        if len(col_data) > sample_size:
            sample_str += ", ..."
        parts.append(f"Sample: [{sample_str}]")

    result = " | ".join(parts)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."

    return result


def _format_value(val, max_str_len: int = 20) -> str:
    """Format a single value for preview display."""
    if val is None:
        return "null"

    if isinstance(val, float):
        if np.isnan(val):
            return "NaN"
        if np.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        # Use reasonable precision
        if abs(val) < 0.001 or abs(val) >= 10000:
            return f"{val:.4e}"
        return f"{val:.4g}"

    if isinstance(val, (int, np.integer)):
        return str(val)

    if isinstance(val, (bytes, np.bytes_)):
        return f"<{len(val)} bytes>"

    if isinstance(val, (np.ndarray,)):
        return f"<array {val.shape}>"

    # String or other
    s = str(val)
    if len(s) > max_str_len:
        return s[:max_str_len - 3] + "..."
    return s
