"""
``ScidbSource`` — the scidb implementation of scistackplot's ``DataSource``.

This is the entire compatibility mechanism: the GUI and every plotting
function above it talk to the protocol, so the same code path serves a lone CSV
(``CsvSource``) and a full scidb project. Nothing above this file knows about
DuckDB, records, or branch params.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
from scistacklog import Log
from scistackplot import (
    LongTable,
    Shape,
    classify_column,
    classify_value,
    is_plottable,
    natural_sort_key,
)
from scistackplot.sources import BaseSource

from .hierarchy import join_frames, joinable, joined_levels
from .load import (
    LATEST_COLUMN,
    data_columns_for,
    load_variable,
    registered_variables,
    sample_value,
    schema_keys,
    variable_levels,
)

LAYER = "scistackplotdb"

#: Column name given to a dict/struct variable's field names when they are
#: melted into long format. Matches the stack's existing vocabulary for "the
#: name of a data column" — ``scidb.ColName()`` and ``PathOutput("{ColName}")``
#: (docs/claude/for-columns-iteration.md) — so a plot faceted by field and a
#: ``for_columns`` run name the same axis the same way.
FIELD_FACTOR = "ColName"


class ScidbSource(BaseSource):
    """
    Plot data straight out of a scidb database.

    ``db`` is an open ``DatabaseManager``. The caller owns it — in the GUI that
    means the ONE manager the server already holds, never a second connection
    (a second DuckDB handle reintroduces the write-lock contention the MATLAB
    run-ownership work resolved).
    """

    def __init__(self, db, *, name: str | None = None) -> None:
        self._db = db
        # `dataset_db_path`, not `db_path` — DatabaseManager has never had the
        # latter, so this silently fell through to "scidb" for every project.
        self.name = name or str(getattr(db, "dataset_db_path", None) or "scidb")
        self._frames: dict[str, Any] = {}
        self._shapes: dict[str, Shape] = {}
        self._levels: dict[str, list[str]] = {}

    # ---- description -----------------------------------------------------

    def describe(self) -> dict:
        """
        Every plottable variable plus the schema keys, without loading data.

        Shapes come from one sampled value per variable (a single-row query),
        so opening the panel on a large database stays cheap; the levels come
        from ``_schema``, which is small by construction.
        """
        with Log.timer("describe", layer=LAYER):
            keys = schema_keys(self._db)
            measures = []
            for variable in registered_variables(self._db):
                shape = self._shape_of(variable)
                columns = data_columns_for(self._db, variable)
                measures.append(
                    {
                        "name": variable,
                        "display": variable,
                        "shape": str(shape),
                        "exploded": False,
                        "plottable": shape
                        in (Shape.SCALAR, Shape.SERIES_1D, Shape.MATRIX_2D),
                        "columns": columns,
                        "levels": self._levels_of(variable),
                    }
                )

            factors = []
            for key in keys:
                levels = self._schema_levels(key)  # one query per key, not two
                factors.append(
                    {
                        "name": key,
                        "display": key,
                        "levels": levels,
                        "level_count": len(levels),
                        "is_variant": False,
                    }
                )

            Log.info(
                "describe: %d variable(s), %d schema key(s)",
                len(measures),
                len(factors),
                layer=LAYER,
            )
            return {
                "name": self.name,
                "schema_keys": keys,
                "factors": factors,
                "measures": measures,
                "index_column": None,
            }

    def _shape_of(self, variable: str) -> Shape:
        if variable not in self._shapes:
            self._shapes[variable] = classify_value(sample_value(self._db, variable))
        return self._shapes[variable]

    def _levels_of(self, variable: str) -> list[str]:
        """Schema depth WITHOUT loading the variable — describe() asks for every
        registered variable, and loading each frame would read the whole
        database to open the panel."""
        if variable in self._frames:
            return self._frames[variable].levels
        if variable not in self._levels:
            self._levels[variable] = variable_levels(self._db, variable)
        return self._levels[variable]

    def _schema_levels(self, key: str) -> list[str]:
        rows = self._db._duck._fetchall(
            f'SELECT DISTINCT s."{key}" FROM _schema s WHERE s."{key}" IS NOT NULL'
        )
        values = [str(row[0]) for row in rows]
        return self._ordered(key, values)

    def _ordered(self, key: str, values: list[str]) -> list[str]:
        """
        Order a factor's levels.

        Declared key types decide, not pandas' default: a key declared
        ``"numeric"`` sorts numerically, and everything else goes through the
        natural-sort key so zero-padded string IDs ("01", "02", … "10") land in
        the order a reader expects instead of the lexicographic 1, 10, 2.
        """
        declared = getattr(self._db, "dataset_schema_key_types", None) or {}
        unique = list(dict.fromkeys(values))
        if declared.get(key) == "numeric":
            def numeric_key(value: str):
                try:
                    return (0, float(value))
                except (TypeError, ValueError):
                    return (1, 0.0)

            return sorted(unique, key=numeric_key)
        return sorted(unique, key=natural_sort_key)

    # ---- data ------------------------------------------------------------

    def _variable_frame(self, variable: str):
        if variable not in self._frames:
            self._frames[variable] = load_variable(self._db, variable)
        return self._frames[variable]

    def get_table(self, measures: list[str]) -> LongTable:
        """
        Build the long table for one or two variables.

        A second measure supplies the x axis of a relational scatter; if the
        two live at different schema depths, the shallower one is broadcast
        down the hierarchy (see :mod:`.hierarchy`).
        """
        if not measures:
            raise ValueError("get_table needs at least one measure (variable name).")
        if len(measures) > 2:
            raise ValueError(
                f"At most 2 measures (y, and optionally x); got {measures}"
            )

        known = registered_variables(self._db)
        unknown = [name for name in measures if name not in known]
        if unknown:
            # Same failure shape as the CSV source's unknown-column error, so
            # callers (and the GUI) handle one kind of "no such measure".
            raise KeyError(f"Unknown variable(s) {unknown}. Available: {known}")

        primary = self._variable_frame(measures[0])
        field_columns: list[str] = []

        if len(measures) == 1:
            if len(primary.data_columns) > 1:
                frame, field_factor = self._melt_fields(primary, measures[0])
                field_columns = [field_factor]
            else:
                frame = self._named_frame(primary, measures[0])
            levels = primary.levels
            variant_columns = list(primary.variant_columns)
            variant_axes = list(primary.variant_axes)
            measure_names = [measures[0]]
            # Open on the current code version.
            latest_column = primary.latest_column
            default_pin = {latest_column: True} if latest_column else None
        else:
            secondary = self._variable_frame(measures[1])
            multi = [
                name
                for name, frame in zip(measures, (primary, secondary), strict=True)
                if len(frame.data_columns) > 1
            ]
            if multi:
                raise ValueError(
                    f"{multi} store one column per dict/struct field, so there is "
                    f"no single value to pair with another measure. Plot one of "
                    f"them on its own (its fields become subplots), or save the "
                    f"field you want as its own variable."
                )
            # Rename BEFORE joining. Two variables' data columns routinely share
            # a name ("value" is the default), and renaming after the merge
            # cannot separate them — one rename key silently shadows the other
            # and the first measure's column vanishes.
            left = replace(primary, frame=self._named_frame(primary, measures[0]))
            right = replace(secondary, frame=self._named_frame(secondary, measures[1]))
            frame = join_frames(
                left,
                right,
                left_value=measures[0],
                right_value=measures[1],
            )
            levels = joined_levels(primary, secondary)
            variant_columns = list(
                dict.fromkeys(primary.variant_columns + secondary.variant_columns)
            )
            variant_axes = list(
                {
                    axis["column"]: axis
                    for axis in primary.variant_axes + secondary.variant_axes
                }.values()
            )
            measure_names = list(measures)
            # `join_frames` now carries both sides' flags through the merge and
            # ANDs them, so the pin-latest default applies to two-measure plots
            # too. It used to drop the flag, which made a relational scatter the
            # one place the default silently stopped protecting the figure.
            latest_column = LATEST_COLUMN if LATEST_COLUMN in frame.columns else None
            default_pin = {latest_column: True} if latest_column else None

        # Variant columns first, and code versions lead within them (see
        # `attach_variants`). The variants are what a reader has to make a
        # decision about — a schema key is just where the data sits — so they
        # get the top of the factor list rather than whatever position the
        # schema happened to leave them.
        factors = [c for c in variant_columns if c in frame.columns]
        factors.extend(key for key in levels if key in frame.columns)
        factors.extend(c for c in field_columns if c in frame.columns)

        level_order = {
            name: self._ordered(name, [str(v) for v in frame[name].dropna().unique()])
            for name in factors
        }

        table = LongTable.from_frame(
            frame,
            factors=factors,
            measures=measure_names,
            level_order=level_order,
            variant_factors=variant_columns,
            field_factors=field_columns,
            name=measures[0],
            default_pin=default_pin,
            latest_column=latest_column,
            factor_origins={axis["column"]: axis for axis in variant_axes},
        )
        Log.debug(
            "get_table(%s): %d row(s), factors=%s",
            measures,
            len(frame),
            factors,
            layer=LAYER,
        )
        return table

    def variant_graph(self, variable: str, functions: list[str] | None = None) -> dict:
        """Variant axes and per-function versions for ``variable``.

        A method rather than a bare function so it reuses this source's frame
        cache: the picker opens over a variable the panel has already loaded,
        and re-reading it to answer "what versions exist" would double the cost
        of opening a dialog.
        """
        from .variants import variant_graph

        return variant_graph(self._db, self._variable_frame(variable), functions)

    def _melt_fields(self, variable_frame, measure: str):
        """
        Turn a dict/struct variable's columns into ONE measure plus a field
        factor.

        scidb stores a dict-valued variable in ``multi_column`` mode — one
        DuckDB column per key (13 muscles of an EMG record become 13 columns;
        see docs/claude/multi-column-save-schema.md). Those columns are
        parallel quantities, not separate variables, so melting them into long
        format makes the field name an ordinary factor. It then gets one
        subplot per level by default (``default_roles``), and the user can move
        it to colour or separate figures like any other factor — which beats
        hardcoding subplots into the renderer.
        """
        frame = variable_frame.frame
        usable, skipped = [], []
        for column in variable_frame.data_columns:
            if is_plottable(classify_column(frame[column])):
                usable.append(column)
            else:
                skipped.append(column)
        if skipped:
            Log.warn(
                "variable %r: %d field(s) are not plottable and were dropped: %s",
                variable_frame.name,
                len(skipped),
                skipped,
                layer=LAYER,
            )
        if not usable:
            raise ValueError(
                f"Variable {variable_frame.name!r} has no plottable fields "
                f"(columns: {variable_frame.data_columns})."
            )

        id_vars = [c for c in frame.columns if c not in variable_frame.data_columns]
        field_factor = FIELD_FACTOR
        while field_factor in id_vars:  # never shadow a schema key
            field_factor += "_"

        melted = frame.melt(
            id_vars=id_vars,
            value_vars=usable,
            var_name=field_factor,
            value_name=measure,
        )
        Log.info(
            "melted %r: %d field(s) -> %d row(s), field factor %r",
            variable_frame.name,
            len(usable),
            len(melted),
            field_factor,
            layer=LAYER,
        )
        return melted, field_factor

    def _named_frame(self, variable_frame, measure: str):
        """The variable's frame with its data column renamed to the measure."""
        return variable_frame.frame.rename(
            columns={self._value_column(variable_frame): measure}
        )

    def _value_column(self, variable_frame) -> str:
        columns = variable_frame.data_columns
        if not columns:
            raise ValueError(
                f"Variable {variable_frame.name!r} has no data column to plot."
            )
        if len(columns) > 1:
            Log.warn(
                "variable %r has %d data columns %s — plotting the first",
                variable_frame.name,
                len(columns),
                columns,
                layer=LAYER,
            )
        return columns[0]

    def joinable_with(self, measure: str) -> list[str]:
        """Variables that can share a plot with ``measure``."""
        own = self._levels_of(measure)
        result = []
        for candidate in registered_variables(self._db):
            if candidate == measure:
                continue
            if self._shape_of(candidate) is not Shape.SCALAR:
                continue  # an x axis must be scalar
            if len(data_columns_for(self._db, candidate)) > 1:
                continue  # a struct has no single value to put on an axis
            if joinable(own, self._levels_of(candidate)):
                result.append(candidate)
        return result

    def default_measure(self) -> str | None:
        for variable in registered_variables(self._db):
            if self._shape_of(variable) in (Shape.SCALAR, Shape.SERIES_1D):
                return variable
        return None

    def invalidate(self, variable: str | None = None) -> None:
        """Drop cached frames after a pipeline run has written new records."""
        if variable is None:
            self._frames.clear()
            self._shapes.clear()
            self._levels.clear()
        else:
            self._frames.pop(variable, None)
            self._shapes.pop(variable, None)
            self._levels.pop(variable, None)
