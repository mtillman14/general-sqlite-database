"""
Code generation: a ``PlotSpec`` becomes readable seaborn/matplotlib source.

Export emits **literal plotting code**, not a call back into this package. The
alternative — ``return scistackplot.render(df, spec_path)`` — is more compact
and stays re-editable in the GUI, but it makes every exported pipeline depend
on this package at runtime and hides the figure's definition behind an opaque
call. Literal code matches the project's "minimize lock-in" goal and the
precedent in ``docs/claude/gui-export-to-plain-python.md``.

The spec is emitted as a docstring block so the GUI can round-trip it back out
of a file the user has since hand-edited.
"""

from __future__ import annotations

import json
import keyword
import re

from .reduce import plan_layout
from .roles import complete_roles
from .shape import Shape
from .spec import ErrorBand, PlotKind, PlotSpec, Role, Statistic
from .table import LongTable
from .variants import (
    LATEST,
    VARIANT_FACTOR,
    apply_variant_sets,
    defined_sets,
    set_name,
)

#: Marker delimiting the embedded spec inside a generated docstring.
SPEC_BEGIN = "scistackplot-spec:"

_SEABORN_ERRORBAR = {
    ErrorBand.SD: '"sd"',
    ErrorBand.SEM: '"se"',
    ErrorBand.CI95: '("ci", 95)',
    ErrorBand.IQR: '("pi", 50)',
    ErrorBand.NONE: "None",
}

_SERIES_COLUMN = "_series"


def generate_plot_function(
    spec: PlotSpec,
    table: LongTable,
    *,
    function_name: str | None = None,
) -> str:
    """
    Generate a ``plot_*`` function body for a scidb endpoint.

    The signature is ``(df, filename)`` — the shape scidb's ``plot_`` contract
    passes — and the function returns a Figure, which the framework saves to
    ``filename`` and closes.
    """
    name = function_name or default_function_name(spec)
    table = apply_variant_sets(spec, table)
    roles = complete_roles(spec, table)
    shape = table.shape_of(spec.y_measure)

    body: list[str] = []
    body.extend(_variant_preamble(spec))
    body.extend(_preamble(spec, table, roles, shape))
    body.extend(_plot_call(spec, table, roles, shape))

    lines = [
        f"def {name}({', '.join(function_params(spec))}):",
        f'    """{_docstring(spec, table, roles)}"""',
        "    import matplotlib.pyplot as plt",
        "    import pandas as pd",
        "    import seaborn as sns",
        "",
    ]
    lines.extend(f"    {line}" if line else "" for line in body)
    return "\n".join(lines) + "\n"


def variant_params(spec: PlotSpec) -> list[tuple[str, str]]:
    """``[(param_name, variant label)]`` — one input per named variant.

    Counted over :func:`~scistackplot.variants.defined_sets` — a row the user
    added but has not filled in yet selects nothing and must not become an
    input.

    Empty below two variants: one variant is a *filter*, fully expressed by the
    ``Variant(...)`` wrapper on the single ``df`` input, and giving it its own
    parameter would rename ``df`` for no gain.

    Two or more is a comparison, and it needs one input each. The endpoint
    cannot receive them as one frame and sort them out afterwards: ``as_table``
    hands a function schema keys and data columns only (``scifor``'s
    ``_extract_data``), never the branch-param or code-version columns that
    distinguish variants — those exist only in ``scistackplotdb``'s loader. So
    the split has to happen where the *load* happens, one input per variant, and
    the labels are re-attached here.
    """
    sets = defined_sets(spec.variant_sets)
    if len(sets) < 2:
        return []
    used: set[str] = set()
    params: list[tuple[str, str]] = []
    for index, variant in enumerate(sets):
        label = set_name(variant, index)
        base = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower() or "variant"
        if base[0].isdigit() or keyword.iskeyword(base):
            base = f"v_{base}"
        candidate, suffix = base, 2
        while candidate in used:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        used.add(candidate)
        params.append((candidate, label))
    return params


def function_params(spec: PlotSpec) -> list[str]:
    """The generated function's signature, in for_each input order."""
    variants = variant_params(spec)
    if not variants:
        return ["df", "filename"]
    return [param for param, _ in variants] + ["filename"]


def _variant_preamble(spec: PlotSpec) -> list[str]:
    """Concatenate the per-variant inputs into one labelled ``df``."""
    variants = variant_params(spec)
    if not variants:
        return []
    lines = [
        "# One input per named variant (each loaded through its own",
        "# Variant(...) filter), labelled and stacked into one frame.",
        "df = pd.concat(",
        "    [",
    ]
    lines.extend(
        f"        {param}.assign(**{{{VARIANT_FACTOR!r}: {label!r}}}),"
        for param, label in variants
    )
    lines.extend(["    ],", "    ignore_index=True,", ")", ""])
    return lines


def default_function_name(spec: PlotSpec) -> str:
    """A valid ``plot_``-prefixed identifier derived from the measure name."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", spec.y_measure).strip("_").lower()
    return f"plot_{slug or 'figure'}"


def extract_spec(source: str) -> PlotSpec | None:
    """
    Recover the embedded spec from generated source.

    Lets the GUI reopen a figure the user has since hand-edited: the code is
    the source of truth for rendering, the embedded spec only for repopulating
    the controls. Returns None when no spec is present.
    """
    start = source.find(SPEC_BEGIN)
    if start == -1:
        return None
    brace = source.find("{", start)
    if brace == -1:
        return None
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return PlotSpec.from_json(source[brace : position + 1])
                except (ValueError, KeyError):
                    return None
    return None


# ---------------------------------------------------------------------------


def _docstring(spec: PlotSpec, table: LongTable, roles: dict) -> str:
    iterate = spec.iterate_factors
    note = ""
    if iterate:
        note = (
            f"\n\n    One figure per {', '.join(iterate)} — these are the "
            f"for_each iteration keys, so they are NOT columns here."
        )
    if spec.facet.has_rules and not _seaborn_can_express_layout(spec, table, roles):
        note += (
            "\n\n    NOTE: the interactive layout arranged the subplots by "
            "matching\n    rules across two grid axes, which seaborn cannot "
            "express — this code\n    wraps them in order instead. The spec "
            "below still carries the rules."
        )
    return (
        f"{spec.kind} of {spec.y_measure}. Generated by scistackplot.{note}\n\n"
        f"    {SPEC_BEGIN}\n    {spec.to_json(indent=None)}\n    "
    )


def _preamble(spec, table, roles, shape) -> list[str]:
    """Melt, filters, 1-D explosion, aggregation — the order resolve() uses."""
    lines: list[str] = []

    # A dict/struct variable arrives at the endpoint as one column per field
    # (scidb's multi_column storage). The interactive path melts it in
    # ScidbSource.get_table, so the generated code has to melt it too — the
    # exported figure must be the previewed figure.
    for field in table.field_factors:
        levels = [str(level) for level in field.levels]
        lines.extend(
            [
                f"# one row per field of {spec.y_measure} "
                f"({len(levels)} field(s))",
                f"_fields = {levels!r}",
                "df = df.melt(",
                "    id_vars=[c for c in df.columns if c not in _fields],",
                "    value_vars=_fields,",
                f"    var_name={field.name!r},",
                f"    value_name={spec.y_measure!r},",
                ")",
                "",
            ]
        )

    filter_lines: list[str] = []
    for flt in spec.filters:
        if flt.include is not None:
            filter_lines.append(f"df = df[df[{flt.column!r}].isin({list(flt.include)!r})]")
        if flt.exclude is not None:
            filter_lines.append(f"df = df[~df[{flt.column!r}].isin({list(flt.exclude)!r})]")
        if flt.minimum is not None:
            filter_lines.append(f"df = df[df[{flt.column!r}] >= {flt.minimum!r}]")
        if flt.maximum is not None:
            filter_lines.append(f"df = df[df[{flt.column!r}] <= {flt.maximum!r}]")
    if filter_lines:
        lines.extend([*filter_lines, ""])

    index_column = spec.index_column or table.index_column or "index"
    if shape is Shape.SERIES_1D and not table.measure(spec.y_measure).exploded:
        y = spec.y_measure
        lines.extend(
            [
                "# 1-D measure: one row per sample",
                f"df[{index_column!r}] = df[{y!r}].map(lambda v: list(range(len(v))))",
                f"df = df.explode([{y!r}, {index_column!r}], ignore_index=True)",
                f"df[{y!r}] = pd.to_numeric(df[{y!r}])",
                f"df[{index_column!r}] = pd.to_numeric(df[{index_column!r}])",
                "",
            ]
        )

    aggregated = [name for name, role in roles.items() if role is Role.AGGREGATE]
    if aggregated:
        keep = [
            name
            for name, role in roles.items()
            if role not in (Role.AGGREGATE, Role.ITERATE)
        ]
        if shape is Shape.SERIES_1D:
            keep.append(index_column)
        lines.extend(
            [
                f"# average over {', '.join(aggregated)}",
                f"df = df.groupby({keep!r}, as_index=False)[{spec.y_measure!r}].mean()",
                "",
            ]
        )

    if spec.kind is PlotKind.LINE:
        series_cols = [
            name
            for name, role in roles.items()
            if role not in (Role.ITERATE, Role.AGGREGATE)
            and name != _role_holder(roles, Role.X)
        ]
        if series_cols:
            lines.extend(
                [
                    "# one line per observation",
                    f"df[{_SERIES_COLUMN!r}] = "
                    f"df[{series_cols!r}].astype(str).agg(' | '.join, axis=1)",
                    "",
                ]
            )

    return lines


def _facet_layout_args(spec, table: LongTable, facets: list[str]) -> list[str]:
    """
    seaborn arguments that reproduce the interactive facet arrangement.

    A single faceted factor is a strip of panels seaborn wraps at ``col_wrap``,
    and the order it wraps them in is ``col_order`` — so a rule-defined layout
    IS expressible whenever the panels fill the grid without holes. Replaying
    ``plan_layout`` here (rather than re-deriving an order) is what keeps the
    exported figure identical to the preview; when the arrangement cannot be
    expressed, ``_docstring`` says so instead of quietly differing.
    """
    if len(facets) != 1:
        return []
    try:
        levels = [str(level) for level in table.factor(facets[0]).levels]
    except KeyError:
        return []
    if not levels:
        return []

    plan = plan_layout(levels, spec.facet)
    args = []
    if plan.n_cols < len(levels):
        args.append(f"col_wrap={plan.n_cols}")
    if spec.facet.has_rules and plan.fills_row_major:
        args.append(f"col_order={plan.labels_in_grid_order(levels)!r}")
    return args


def _seaborn_can_express_layout(spec, table: LongTable, roles) -> bool:
    """Whether ``_facet_layout_args`` reproduced the rules (see ``_docstring``)."""
    facets = [name for name, role in roles.items() if role is Role.FACET]
    if len(facets) != 1:
        return False
    try:
        levels = [str(level) for level in table.factor(facets[0]).levels]
    except KeyError:
        return False
    return bool(levels) and plan_layout(levels, spec.facet).fills_row_major


def _plot_call(spec, table, roles, shape) -> list[str]:
    if shape is Shape.MATRIX_2D:
        return _heatmap_call(spec)

    kind = spec.kind
    x = _x_expression(spec, table, roles, shape)
    color = _role_holder(roles, Role.COLOR)
    facets = [name for name, role in roles.items() if role is Role.FACET]

    args = [f"data=df", f"x={x!r}", f"y={spec.y_measure!r}"]
    if color:
        args.append(f"hue={color!r}")
        if _color_level_count(spec, table, color) < 2:
            # Same rule the renderers apply (render.base.shows_legend): one
            # colour level means the legend restates what every mark on the
            # figure has in common. The exported figure must be the previewed
            # figure, so it has to be decided here too, not just at render time.
            args.append("legend=False")
    # seaborn takes one factor per grid axis: the first faceted factor drives
    # the columns, a second one the rows.
    if facets:
        args.append(f"col={facets[0]!r}")
    if len(facets) > 1:
        args.append(f"row={facets[1]!r}")
    args.extend(_facet_layout_args(spec, table, facets))

    estimator = (
        '"median"' if spec.aggregate.statistic is Statistic.MEDIAN else '"mean"'
    )
    errorbar = _SEABORN_ERRORBAR[spec.aggregate.error]

    if kind in (PlotKind.BOX, PlotKind.VIOLIN, PlotKind.BAR, PlotKind.STRIP):
        seaborn_kind = {
            PlotKind.BOX: "box",
            PlotKind.VIOLIN: "violin",
            PlotKind.BAR: "bar",
            PlotKind.STRIP: "strip",
        }[kind]
        args.append(f'kind="{seaborn_kind}"')
        if kind is PlotKind.BAR:
            args.append(f"estimator={estimator}")
            args.append(f"errorbar={errorbar}")
        call = "sns.catplot"
    elif kind is PlotKind.SCATTER:
        if _x_is_categorical(spec, table, roles, shape):
            args.extend(['kind="strip"', "jitter=False"])
            call = "sns.catplot"
        else:
            args.append('kind="scatter"')
            call = "sns.relplot"
    elif kind is PlotKind.LINE:
        args.append('kind="line"')
        args.append("estimator=None")
        if any(role is Role.FREE for role in roles.values()):
            args.append(f"units={_SERIES_COLUMN!r}")
        call = "sns.relplot"
    elif kind is PlotKind.BAND:
        args.extend([f'kind="line"', f"estimator={estimator}", f"errorbar={errorbar}"])
        call = "sns.relplot"
    else:  # pragma: no cover - every kind is covered above
        args.append('kind="scatter"')
        call = "sns.relplot"

    style = spec.style
    if style.palette:
        args.append(f"palette={style.palette!r}")

    lines = [f"g = {call}(", *[f"    {arg}," for arg in args], ")"]
    lines.append(
        f"g.set_axis_labels({(style.x_label or x)!r}, "
        f"{(style.y_label or spec.y_measure)!r})"
    )
    if style.log_x:
        lines.append('g.set(xscale="log")')
    if style.log_y:
        lines.append('g.set(yscale="log")')
    if style.title:
        lines.append(f"g.figure.suptitle({style.title!r})")
    lines.append(f"g.figure.set_size_inches({style.width}, {style.height})")
    lines.append("return g.figure")
    return lines


def _heatmap_call(spec) -> list[str]:
    return [
        "import numpy as np",
        "",
        f"matrix = np.mean(np.stack([np.asarray(v, dtype=float) "
        f"for v in df[{spec.y_measure!r}]]), axis=0)",
        f"fig, ax = plt.subplots(figsize=({spec.style.width}, {spec.style.height}))",
        'image = ax.imshow(matrix, aspect="auto", origin="lower")',
        "fig.colorbar(image, ax=ax)",
        f"ax.set_title({(spec.style.title or spec.y_measure)!r})",
        "return fig",
    ]


def _color_level_count(spec: PlotSpec, table: LongTable, color: str) -> int:
    """
    How many colour levels the generated code will actually draw.

    Filters are applied here because they are applied in the generated
    preamble: filtering a two-level factor down to one must drop the legend in
    the export exactly as it drops it in the preview. An unknown factor keeps
    its legend — omitting one that was wanted is worse than keeping one that
    was not.
    """
    try:
        levels = [str(level) for level in table.factor(color).levels]
    except KeyError:
        return 2
    for flt in spec.filters:
        if flt.column != color:
            continue
        if flt.include is not None:
            keep = {str(value) for value in flt.include}
            levels = [level for level in levels if level in keep]
        if flt.exclude is not None:
            drop = {str(value) for value in flt.exclude}
            levels = [level for level in levels if level not in drop]
    return len(levels)


def _x_expression(spec, table, roles, shape) -> str:
    if spec.x_measure:
        return spec.x_measure
    if shape is Shape.SERIES_1D:
        return spec.index_column or table.index_column or "index"
    holder = _role_holder(roles, Role.X)
    return holder or (table.factor_names[0] if table.factors else "index")


def _x_is_categorical(spec, table, roles, shape) -> bool:
    if spec.x_measure or shape is Shape.SERIES_1D:
        return False
    return _role_holder(roles, Role.X) is not None


def _role_holder(roles: dict[str, Role], role: Role) -> str | None:
    for name, assigned in roles.items():
        if assigned is role:
            return name
    return None


def generate_script(
    spec: PlotSpec,
    table: LongTable,
    *,
    source_expression: str = 'pd.read_csv("data.csv")',
    function_name: str | None = None,
) -> str:
    """
    A complete runnable script — the standalone (no scidb) export.

    Same generated function, plus the few lines that load a table and save the
    figure, so a CSV user gets something they can run immediately.
    """
    name = function_name or default_function_name(spec)
    function = generate_plot_function(spec, table, function_name=name)
    call_args, setup = _script_inputs(spec)
    return (
        '"""Generated by scistackplot."""\n'
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n\n"
        f"{function}\n\n"
        'if __name__ == "__main__":\n'
        f"    df = {source_expression}\n"
        + "".join(f"    {line}\n" if line else "\n" for line in setup)
        + f'    figure = {name}({call_args}"figure.png")\n'
        '    figure.savefig("figure.png", dpi=150, bbox_inches="tight")\n'
    )


def _script_inputs(spec: PlotSpec) -> tuple[str, list[str]]:
    """The standalone script's per-variant frames.

    The endpoint path gets its variants from the database, one ``Variant(...)``
    load per input. A script has one flat frame instead, so the same split is
    done here with literal ``pandas`` — the variant columns are ordinary columns
    of whatever was loaded.

    A ``"latest"`` selection cannot be honoured standalone: which record is
    newest at a schema location is a provenance question, and a CSV carries no
    provenance. Rather than silently pinning nothing, the generated line says so.
    """
    variants = variant_params(spec)
    if not variants:
        return "df, ", []

    lines: list[str] = [""]
    for (param, label), variant in zip(
        variants, defined_sets(spec.variant_sets), strict=True
    ):
        lines.append(f"# variant {label!r}")
        lines.append(f"{param} = df")
        for column, value in variant.selection.items():
            if isinstance(value, str) and value == LATEST:
                lines.append(
                    f"# NOTE: {column!r} asked for the latest version, which "
                    f"needs provenance a flat table does not carry — not applied."
                )
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                levels = [str(v) for v in value]
                test = f"{param}[{column!r}].astype(str).isin({levels!r})"
            else:
                test = f"{param}[{column!r}].astype(str) == {str(value)!r}"
            lines.append(f"{param} = {param}[{test}]")
    lines.append("")
    return "".join(f"{param}, " for param, _ in variants), lines


def spec_json_block(spec: PlotSpec) -> str:
    """The spec as a pretty JSON block, for writing next to generated code."""
    return json.dumps(spec.to_dict(), indent=2, sort_keys=True)
