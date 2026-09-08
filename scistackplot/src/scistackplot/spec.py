"""
``PlotSpec`` — the serializable description of a plot.

This is the load-bearing object of the whole package. The interactive GUI does
not produce pictures; it produces a ``PlotSpec``, and everything downstream
(resolution, rendering, code generation, and the pipeline ``plot_`` endpoint)
is a pure function of one. Keeping it small, serializable, and diffable is what
lets an inherently visual tool participate in a reproducible pipeline.

See ``docs/claude/plotting-library-design.md`` for the reasoning.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class Role(str, Enum):
    """
    What a factor column does in a plot.

    Every factor carries exactly **one** role. That invariant is the whole
    reason this enum exists: the R/Shiny proof of concept spread the same
    information across four independent widgets and spent most of its length
    keeping them consistent with each other via ``setdiff``.
    """

    ITERATE = "iterate"          # separate FIGURE per level (fan-out)
    X = "x"                      # x-axis position
    COLOR = "color"              # one series/hue per level
    FACET = "facet"              # one subplot per level
    AGGREGATE = "aggregate"      # collapse: average across this factor's levels
    FREE = "free"                # keep as replicate rows -> distributions

    def __str__(self) -> str:
        return self.value


#: Roles that accept at most ONE factor. FACET is deliberately absent: several
#: factors may be faceted at once, and how their combined levels are arranged
#: into rows and columns is a layout decision (FacetOptions.rows/cols), not a
#: property of which factor was assigned where. The old FACET_ROW/FACET_COL
#: pair forced that decision into the role and still could not express
#: "arrange these 13 muscles as left/right x muscle group".
SINGLE_ASSIGNMENT_ROLES = (Role.X, Role.COLOR)

#: Roles that leave a factor's levels as multiple rows in one cell, i.e. that
#: can produce a distribution. AGGREGATE is NOT here: it collapses first.
REPLICATE_ROLES = (Role.FREE,)


class PlotKind(str, Enum):
    """The visual form of the plot."""

    SCATTER = "scatter"      # one marker per row
    STRIP = "strip"          # scatter with categorical jitter
    LINE = "line"            # one polyline per series (1-D measures)
    BOX = "box"              # distribution per x position
    VIOLIN = "violin"        # distribution per x position, density
    BAR = "bar"              # statistic per x position, with error bar
    BAND = "band"            # statistic line + shaded error region (1-D)
    HEATMAP = "heatmap"      # 2-D matrix

    def __str__(self) -> str:
        return self.value


class Statistic(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"

    def __str__(self) -> str:
        return self.value


class ErrorBand(str, Enum):
    NONE = "none"
    SD = "sd"
    SEM = "sem"
    CI95 = "ci95"
    IQR = "iqr"

    def __str__(self) -> str:
        return self.value


class VariantPolicy(str, Enum):
    """
    What to do when the source supplies branch-param (variant) factors.

    ``POOL`` must be chosen deliberately. Pooling variants silently plots two
    different pipelines' results as if they were replicates of one — a figure
    that is wrong in a way that looks like data. See the design doc.
    """

    FACET = "facet"   # variants become an ordinary factor the user assigns
    PIN = "pin"       # keep only the pinned variant (see PlotSpec.pinned_variant)
    POOL = "pool"     # explicitly average across variants

    def __str__(self) -> str:
        return self.value


class MatchOp(str, Enum):
    """How a facet-layout rule tests a panel's label."""

    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EQUALS = "equals"
    REGEX = "regex"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Matcher:
    """
    One row (or column) of a facet grid, defined by what its panels look like.

    Naming a grid by *rules* rather than by position is what makes a layout
    reusable: "left column = names starting with L" holds for any muscle set,
    any subject, any variable, so the same arrangement can later be saved as a
    preset and applied to a different plot.
    """

    op: MatchOp = MatchOp.CONTAINS
    value: str = ""
    #: Shown as the row/column header. Defaults to the value.
    label: str | None = None

    @property
    def display(self) -> str:
        """Row/column header. An unset slot has no header, not the op's name."""
        if self.is_blank:
            return self.label or ""
        return self.label or self.value

    @property
    def is_blank(self) -> bool:
        """
        An unset slot. The grid has a fixed number of row/column slots, so most
        of them are empty most of the time, and an empty slot must mean
        "whatever is left over, in order" — never "everything".
        """
        return not self.value

    def matches(self, text: str) -> bool:
        if self.is_blank:
            # Without this, CONTAINS "" matches every panel ('' in text) and the
            # first blank slot swallows the whole grid; NOT_CONTAINS "" is the
            # same bug inverted.
            return False
        text = "" if text is None else str(text)
        needle = self.value
        if self.op is MatchOp.STARTS_WITH:
            return text.startswith(needle)
        if self.op is MatchOp.ENDS_WITH:
            return text.endswith(needle)
        if self.op is MatchOp.CONTAINS:
            return needle in text
        if self.op is MatchOp.NOT_CONTAINS:
            return needle not in text
        if self.op is MatchOp.EQUALS:
            return text == needle
        if self.op is MatchOp.REGEX:
            import re

            try:
                return re.search(needle, text) is not None
            except re.error:
                # An invalid pattern must not take the whole figure down while
                # the user is still typing it.
                return False
        return False


@dataclass(frozen=True)
class Filter:
    """A row filter applied before anything else."""

    column: str
    #: Keep rows whose value is in this list (categorical include).
    include: list[Any] | None = None
    #: Drop rows whose value is in this list.
    exclude: list[Any] | None = None
    #: Numeric range, inclusive; either bound may be None.
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Aggregation:
    statistic: Statistic = Statistic.MEAN
    error: ErrorBand = ErrorBand.SD


@dataclass(frozen=True)
class FacetOptions:
    """
    How faceted panels are arranged: a grid of a known size, plus one ordering
    rule per row and per column slot.

    ``n_rows``/``n_cols`` size the grid; setting either one computes the other
    from the panel count (:func:`grid_shape_for`), so the user names a width and
    the height follows. Each slot may then carry a :class:`Matcher` that claims
    the panels whose label it matches, which is what lets an EMG figure read as
    left/right x muscle group instead of an arbitrary 4-wide flow. Blank slots
    take whatever is left over, in order.
    """

    #: Grid size. None means "compute me from the other one and the panel count".
    n_rows: int | None = None
    n_cols: int | None = None
    #: One entry per row / column slot; blank entries are unset (see Matcher.is_blank).
    rows: list[Matcher] = field(default_factory=list)
    cols: list[Matcher] = field(default_factory=list)
    share_x: bool = True
    share_y: bool = True

    @property
    def row_rules(self) -> list[Matcher]:
        """Row slots that actually claim panels."""
        return [m for m in self.rows if not m.is_blank]

    @property
    def col_rules(self) -> list[Matcher]:
        return [m for m in self.cols if not m.is_blank]

    @property
    def has_rules(self) -> bool:
        return bool(self.row_rules or self.col_rules)


@dataclass(frozen=True)
class StyleOptions:
    palette: str | None = None
    width: float = 8.0
    height: float = 6.0
    log_x: bool = False
    log_y: bool = False
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    marker_size: float = 36.0
    alpha: float = 0.85


@dataclass(frozen=True)
class PlotSpec:
    """
    A complete, serializable plot description.

    ``measures[0]`` is always the y measure. A second measure, when present,
    supplies x for an x–y scatter (in which case no factor may hold ``Role.X``).
    """

    measures: list[str]
    roles: dict[str, Role] = field(default_factory=dict)
    kind: PlotKind = PlotKind.SCATTER
    aggregate: Aggregation = field(default_factory=Aggregation)
    index_column: str | None = None
    facet: FacetOptions = field(default_factory=FacetOptions)
    style: StyleOptions = field(default_factory=StyleOptions)
    filters: list[Filter] = field(default_factory=list)
    variant_policy: VariantPolicy = VariantPolicy.FACET
    #: ``{factor: level}`` or ``{factor: [level, ...]}`` — which variant levels
    #: survive under ``VariantPolicy.PIN``.
    #:
    #: A list means "keep any of these", which is what makes the pin a *subcube*
    #: selector rather than a single-point one. That is the whole difference
    #: between answering "show me one variant" and "show me every variant where
    #: bandpass=v1" — the second is the question a multi-layer project actually
    #: asks, and it cannot be expressed one scalar at a time.
    pinned_variant: dict[str, Any] | None = None

    # ---- convenience accessors ------------------------------------------

    @property
    def y_measure(self) -> str:
        return self.measures[0]

    @property
    def x_measure(self) -> str | None:
        return self.measures[1] if len(self.measures) > 1 else None

    def factors_with_role(self, role: Role) -> list[str]:
        """Factors carrying ``role``, in the spec's declared order."""
        return [name for name, r in self.roles.items() if r == role]

    def first_with_role(self, role: Role) -> str | None:
        found = self.factors_with_role(role)
        return found[0] if found else None

    @property
    def iterate_factors(self) -> list[str]:
        return self.factors_with_role(Role.ITERATE)

    @property
    def replicate_factors(self) -> list[str]:
        """Factors whose levels survive as multiple rows per cell."""
        return [n for n, r in self.roles.items() if r in REPLICATE_ROLES]

    def with_roles(self, **roles: Role) -> "PlotSpec":
        """Return a copy with role assignments merged in (for tests/GUI edits)."""
        merged = dict(self.roles)
        merged.update(roles)
        return replace(self, roles=merged)

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["roles"] = {k: str(v) for k, v in self.roles.items()}
        raw["kind"] = str(self.kind)
        raw["variant_policy"] = str(self.variant_policy)
        raw["aggregate"] = {
            "statistic": str(self.aggregate.statistic),
            "error": str(self.aggregate.error),
        }
        raw["facet"] = {
            "n_rows": self.facet.n_rows,
            "n_cols": self.facet.n_cols,
            "share_x": self.facet.share_x,
            "share_y": self.facet.share_y,
            "rows": [_matcher_to_dict(m) for m in self.facet.rows],
            "cols": [_matcher_to_dict(m) for m in self.facet.cols],
        }
        # TOML has no null; drop empty optionals so a round trip is stable.
        return _drop_nulls(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PlotSpec":
        agg = raw.get("aggregate") or {}
        return cls(
            measures=list(raw["measures"]),
            roles={k: Role(v) for k, v in (raw.get("roles") or {}).items()},
            kind=PlotKind(raw.get("kind", PlotKind.SCATTER)),
            aggregate=Aggregation(
                statistic=Statistic(agg.get("statistic", Statistic.MEAN)),
                error=ErrorBand(agg.get("error", ErrorBand.SD)),
            ),
            index_column=raw.get("index_column"),
            facet=_facet_from_dict(raw.get("facet") or {}),
            style=StyleOptions(**(raw.get("style") or {})),
            filters=[Filter(**f) for f in (raw.get("filters") or [])],
            variant_policy=VariantPolicy(
                raw.get("variant_policy", VariantPolicy.FACET)
            ),
            pinned_variant=raw.get("pinned_variant"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PlotSpec":
        return cls.from_dict(json.loads(text))

    def to_toml(self) -> str:
        try:
            import tomli_w
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Writing a PlotSpec as TOML needs 'tomli-w' "
                "(pip install scistackplot[dev]). PlotSpec.to_json() is "
                "always available."
            ) from exc
        return tomli_w.dumps(self.to_dict())

    @classmethod
    def from_toml(cls, text: str) -> "PlotSpec":
        try:
            import tomllib
        except ImportError:  # pragma: no cover - Python 3.10
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "Reading a PlotSpec from TOML needs Python 3.11+ "
                    "(tomllib) or the 'tomli' package."
                ) from exc
        return cls.from_dict(tomllib.loads(text))


def grid_shape_for(
    n_panels: int, n_rows: int | None = None, n_cols: int | None = None
) -> tuple[int, int]:
    """
    The subplot grid for ``n_panels`` panels, given whatever the user pinned.

    One function, called from ``reduce`` and reported back to the GUI, so that
    "I said 2 columns, where did 3 rows come from?" has exactly one answer.
    Naming one dimension computes the other; naming neither falls back to a
    roughly square grid at most 4 wide, and a handful of panels stay in a single
    horizontal row.

    Placement may still grow the result (see ``reduce._assign_grid``) rather than
    let two panels share a cell.
    """
    count = max(1, int(n_panels))
    rows = int(n_rows) if n_rows and n_rows > 0 else None
    cols = int(n_cols) if n_cols and n_cols > 0 else None

    if rows and cols:
        return rows, cols
    if cols:
        return math.ceil(count / cols), cols
    if rows:
        return rows, math.ceil(count / rows)

    # Neither pinned: a 13-field struct wants a grid, 3 muscles want one row.
    auto_cols = min(4, math.ceil(math.sqrt(count))) if count > 3 else count
    auto_cols = max(1, auto_cols)
    return math.ceil(count / auto_cols), auto_cols


def _matcher_to_dict(matcher: Matcher) -> dict:
    return {"op": str(matcher.op), "value": matcher.value, "label": matcher.label}


def _facet_from_dict(raw: dict) -> FacetOptions:
    return FacetOptions(
        n_rows=raw.get("n_rows"),
        n_cols=raw.get("n_cols"),
        rows=[_matcher_from_dict(m) for m in (raw.get("rows") or [])],
        cols=[_matcher_from_dict(m) for m in (raw.get("cols") or [])],
        share_x=raw.get("share_x", True),
        share_y=raw.get("share_y", True),
    )


def _matcher_from_dict(raw: dict) -> Matcher:
    return Matcher(
        op=MatchOp(raw.get("op", MatchOp.CONTAINS)),
        value=raw.get("value", ""),
        label=raw.get("label"),
    )


def _drop_nulls(obj: Any) -> Any:
    """Recursively drop None values so JSON and TOML round-trip identically."""
    if isinstance(obj, dict):
        return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_nulls(v) for v in obj]
    return obj
