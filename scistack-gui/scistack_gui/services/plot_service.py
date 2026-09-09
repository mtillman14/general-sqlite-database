"""
Plot Studio backend.

One service, two transports: the FastAPI routes in ``api/plot.py`` and the
JSON-RPC handlers in ``server.py`` both call these functions, so the web GUI
and the VS Code extension can never drift apart.

Everything plot-related that is *policy* — which kinds are available, what the
default roles are, how data is reduced — lives in ``scistackplot`` /
``scistackplotdb`` (CLAUDE.md NOTE 3). This module only adapts: it turns JSON
payloads into specs, holds the source cache, and hands results back JSON-safe.

The database handle is the ONE the GUI already owns. A second DuckDB
connection would reintroduce the write-lock contention the MATLAB
run-ownership work resolved — see docs/claude/matlab-run-database-ownership.md.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Cached sources, keyed by ``("db", <database file path>)`` or
#: ``("csv", <path>)``.
#:
#: A run that writes records makes a db source stale — it caches whole frames,
#: so a stale one serves PRE-RUN data and the user plots the wrong thing. That
#: is what ``invalidate()`` is for; ``api.run._notify_records_changed`` calls it
#: wherever a run announces ``dag_updated``.
#:
#: Keyed by path rather than by ``id(db)``: CPython reuses ids after garbage
#: collection, so a freshly-opened DatabaseManager could land on a dead entry's
#: key and inherit another database's cached frames.
_sources: dict[tuple, Any] = {}


def _database_path(db) -> str:
    """The database file a source is keyed on.

    Accepts a ``DatabaseManager`` or a plain path, so callers that hold no
    connection (the MATLAB run threads release it to the sidecar) can still
    name the cache entry.
    """
    return str(getattr(db, "dataset_db_path", None) or db)


def _require_scistackplot():
    """Import the plotting packages, with an actionable message if missing."""
    try:
        import scistackplot  # noqa: F401
        import scistackplotdb  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Plotting needs the scistackplot and scistackplotdb packages: "
            "pip install scistackplot scistackplotdb"
        ) from exc


def get_source(db, *, refresh: bool = False, csv_path: str | None = None):
    """
    The cached source for this request.

    ``csv_path`` selects the standalone CSV implementation instead of the
    scidb one. Both satisfy the same ``DataSource`` protocol, so nothing below
    this line changes — which is the point: the CSV path stays a first-class
    entry point (right-click a .csv in the Explorer, no project, no database)
    rather than a degraded mode, and keeps the standalone claim exercised.
    """
    _require_scistackplot()

    if csv_path:
        from scistackplot import CsvSource

        key = ("csv", csv_path)
        if refresh or key not in _sources:
            _sources[key] = CsvSource(csv_path)
            logger.info("[plot] built CsvSource for %s", csv_path)
        return _sources[key]

    from scistackplotdb import ScidbSource

    key = ("db", _database_path(db))
    if refresh or key not in _sources:
        _sources[key] = ScidbSource(db)
        logger.info("[plot] built ScidbSource for %s (refresh=%s)", key[1], refresh)
    return _sources[key]


def invalidate(db=None) -> dict:
    """Drop cached frames — call after a run may have written records.

    ``db`` may be a ``DatabaseManager`` or a plain path; None drops every
    cached source. Dropping a source that was never built is a no-op, so this
    is safe to call unconditionally after any run.
    """
    if db is None:
        dropped = len(_sources)
        _sources.clear()
    else:
        dropped = 1 if _sources.pop(("db", _database_path(db)), None) else 0
    logger.info(
        "[plot] source cache invalidated (%s, %d source(s) dropped)",
        "all" if db is None else _database_path(db),
        dropped,
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------


def describe(
    db,
    variable: str | None = None,
    *,
    refresh: bool = False,
    csv_path: str | None = None,
) -> dict:
    """
    Everything the panel needs to open on a variable.

    With no ``variable``, returns just the catalog (what is plottable at all).
    With one, adds that variable's factors, a default spec, and the capability
    report — one round trip from "user right-clicked a node" to a rendered
    panel.
    """
    from scistackplot import capabilities, default_spec

    source = get_source(db, refresh=refresh, csv_path=csv_path)
    catalog = source.describe()

    # Entry points that name no variable (the CSV command, the palette with an
    # empty box) open on the first plottable measure rather than on nothing.
    if not variable:
        variable = source.default_measure()
    if not variable:
        return {"catalog": catalog, "variable": None}

    table = source.get_table([variable])
    shape = table.shape_of(variable)
    if not _plottable(shape):
        return {
            "catalog": catalog,
            "variable": variable,
            "eligible": False,
            "reason": _ineligible_reason(shape, len(table.frame)),
            "table": table.describe(),
        }

    # default_spec owns roles + kind + facet wrap together: a 13-field struct
    # opens as a wrapped grid of subplots, not one overplotted axis.
    spec = default_spec(table, variable)

    return {
        "catalog": catalog,
        "variable": variable,
        "eligible": True,
        "reason": None,
        "table": table.describe(),
        "spec": spec.to_dict(),
        "capabilities": capabilities(spec, table),
        "joinable_with": source.joinable_with(variable),
    }


def _plottable(shape) -> bool:
    from scistackplot import Shape

    return shape in (Shape.SCALAR, Shape.SERIES_1D, Shape.MATRIX_2D)


def _ineligible_reason(shape, row_count: int) -> str:
    from scistackplot import Shape

    if row_count == 0:
        # The empty state the design doc insists on naming explicitly: a
        # variable whose pipeline has never run must say so, not draw blank axes.
        return "This variable has no records yet — run the pipeline first."
    if shape is Shape.CATEGORICAL:
        return "This variable holds text, which has no numeric axis to plot."
    return f"Not plottable: values classify as {shape}."


# ---------------------------------------------------------------------------
# Capabilities and resolution
# ---------------------------------------------------------------------------


def capabilities_for(db, spec_payload: dict, *, csv_path: str | None = None) -> dict:
    """Recompute available plot kinds after the user moves a factor's role."""
    from scistackplot import capabilities

    source = get_source(db, csv_path=csv_path)
    spec = _spec_from_payload(spec_payload)
    table = source.get_table(list(spec.measures))
    return capabilities(spec, table)


def variant_graph(
    db,
    variable: str,
    *,
    functions: list[str] | None = None,
    csv_path: str | None = None,
) -> dict:
    """
    Variant axes and per-function versions, for the variant-selection popup.

    ``functions`` is every function node currently on the canvas: the popup
    mirrors the whole pipeline, so a node outside this variable's chain still
    has to be able to say what versions it has run (and, by their absence, that
    selecting one would not affect this figure).

    A CSV has no provenance and therefore no variants — it returns the empty
    graph rather than an error, so the popup can open and say so.
    """
    source = get_source(db, csv_path=csv_path)
    if csv_path or not hasattr(source, "variant_graph"):
        return {"variable": variable, "axes": [], "versions": {}, "chain_functions": []}
    graph = source.variant_graph(variable, functions or [])
    logger.info(
        "[plot] variant_graph(%s): %d axes, %d function(s) with versions",
        variable,
        len(graph["axes"]),
        len(graph["versions"]),
    )
    return graph


def resolve_figures(
    db,
    spec_payload: dict,
    *,
    max_points: int | None = None,
    csv_path: str | None = None,
) -> dict:
    """
    Reduce and render for the interactive panel.

    Returns plotly.js figure dicts — plain JSON, built without the plotly
    package. Downsampling is applied here and only here: the export path
    (``export_code``) never reduces data for transport.
    """
    from scistackplot import MAX_TRANSPORT_POINTS, RoleError, render_plotly, resolve

    source = get_source(db, csv_path=csv_path)
    spec = _spec_from_payload(spec_payload)
    table = source.get_table(list(spec.measures))

    budget = MAX_TRANSPORT_POINTS if max_points is None else max_points
    try:
        resolved = resolve(spec, table, max_points=budget)
    except RoleError as exc:
        # A role conflict is a user-correctable state, not a server fault: the
        # panel shows the message (which names the one-line fix) instead of an
        # error toast.
        logger.info("[plot] invalid spec: %s", exc)
        return {"ok": False, "error": str(exc), "figures": []}

    figures = [
        {
            "key": item.to_dict()["figure_key"],
            "label": item.figure_label,
            "figure": render_plotly(item),
            "row_count": item.row_count,
            "downsampled_from": item.downsampled_from,
        }
        for item in resolved
    ]
    logger.info(
        "[plot] resolved %s: %d figure(s), %d row(s)",
        spec.kind,
        len(figures),
        sum(f["row_count"] for f in figures),
    )
    return {"ok": True, "error": None, "figures": figures}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_code(
    db,
    spec_payload: dict,
    *,
    function_name: str | None = None,
    output_variable: str | None = None,
    path_template: str | None = None,
    finalized: bool = True,
    csv_path: str | None = None,
) -> dict:
    """Generate the ``plot_`` function and its ``for_each`` call, without writing."""
    from scistackplotdb import generate_endpoint

    source = get_source(db, csv_path=csv_path)
    spec = _spec_from_payload(spec_payload)
    table = source.get_table(list(spec.measures))

    code = generate_endpoint(
        spec,
        table,
        input_variable=spec.measures[0],
        x_variable=spec.measures[1] if len(spec.measures) > 1 else None,
        function_name=function_name,
        output_variable=output_variable,
        path_template=path_template,
        finalized=finalized,
    )
    return {
        "ok": True,
        "function_name": code.function_name,
        "function_source": code.function_source,
        "foreach_source": code.foreach_source,
        "source": code.source,
        "iterate_keys": code.iterate_keys,
        "path_template": code.path_template,
        "output_variable": code.output_variable,
    }


def save_figure(
    db,
    spec_payload: dict,
    path: str,
    *,
    dpi: int = 200,
    csv_path: str | None = None,
) -> dict:
    """
    Render the current spec to an image file.

    Rendered with **matplotlib**, not by asking the browser to download the
    plotly view. Two reasons: a webview cannot save a file (its download is
    blocked, which is what made plotly's camera button fail silently), and the
    matplotlib renderer is the one the pipeline uses — so the file you save by
    hand is the same figure ``for_each`` would produce from the exported code.

    Deliberately NO downsampling (``max_points=None``): the interactive view is
    reduced for transport, a saved figure must not be.
    """
    from pathlib import Path

    from scistackplot import RoleError, render_matplotlib, resolve

    source = get_source(db, csv_path=csv_path)
    spec = _spec_from_payload(spec_payload)
    table = source.get_table(list(spec.measures))

    try:
        resolved = resolve(spec, table)
    except RoleError as exc:
        return {"ok": False, "error": str(exc), "files": []}

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"

    written: list[str] = []
    for item in resolved:
        # A spec with ITERATE roles is several figures; give each its own file
        # rather than silently saving only the first.
        if len(resolved) > 1:
            slug = _slug(item.figure_label) or f"{len(written) + 1}"
            out = target.with_name(f"{target.stem}_{slug}{suffix}")
        else:
            out = target.with_suffix(suffix)

        figure = render_matplotlib(item)
        try:
            figure.savefig(out, dpi=dpi, bbox_inches="tight")
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        written.append(str(out))

    logger.info("[plot] saved %d figure(s): %s", len(written), written)
    return {"ok": True, "error": None, "files": written}


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def add_to_pipeline(
    db,
    spec_payload: dict,
    *,
    function_name: str | None = None,
    output_variable: str | None = None,
    path_template: str | None = None,
    finalized: bool = True,
) -> dict:
    """
    Write the generated endpoint into the project and refresh the registry.

    The function is appended to ``scistack_plots.py`` beside the entities file
    — the project root, which is inside the discovery scope — and the output
    variable type is declared through the normal entity-creation path, so a
    generated endpoint never references an undeclared type.

    The ``for_each`` call itself is returned rather than written: the GUI
    builds pipelines from the DAG, so the user wires the newly discovered
    function up on the canvas (or pastes the snippet into a script).
    """
    from pathlib import Path

    from scistack_gui.services.target_file_service import (
        _reload_after_write,
        get_or_create_target_file,
        validate_entity_name,
    )
    from scistack_gui.services.variable_service import create_variable

    generated = export_code(
        db,
        spec_payload,
        function_name=function_name,
        output_variable=output_variable,
        path_template=path_template,
        finalized=finalized,
    )

    name_error = validate_entity_name(generated["output_variable"])
    if name_error:
        return {"ok": False, "error": name_error}

    declared = create_variable(generated["output_variable"])
    if not declared.get("ok") and "already exists" not in str(declared.get("error", "")):
        return {"ok": False, "error": declared.get("error")}

    target_file, target_err = get_or_create_target_file()
    if target_file is None:
        return {"ok": False, "error": target_err}

    plots_file = Path(target_file).parent / "scistack_plots.py"
    existing = plots_file.read_text(encoding="utf-8") if plots_file.exists() else ""
    if f"def {generated['function_name']}(" in existing:
        return {
            "ok": False,
            "error": (
                f"{plots_file.name} already defines {generated['function_name']}. "
                f"Choose a different function name, or edit the existing one."
            ),
        }

    header = '"""Plot endpoints generated by scistackplot."""\n' if not existing else ""
    with open(plots_file, "a", encoding="utf-8") as handle:
        handle.write(header)
        handle.write("\n\n" if existing else "\n")
        handle.write(generated["function_source"])

    logger.info(
        "[plot] wrote %s to %s (output=%s)",
        generated["function_name"],
        plots_file,
        generated["output_variable"],
    )
    reload_error = _reload_after_write(plots_file)
    if reload_error:
        return {**generated, "ok": False, "error": reload_error, "file": str(plots_file)}

    return {**generated, "file": str(plots_file)}


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _spec_from_payload(payload: dict):
    from scistackplot import PlotSpec

    _require_scistackplot()
    if not payload or not payload.get("measures"):
        raise ValueError("A plot spec needs at least one measure (variable name).")
    return PlotSpec.from_dict(payload)

