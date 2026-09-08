"""
Plot Studio backend (services/plot_service.py).

The service is an adapter: policy lives in scistackplot/scistackplotdb, and
these tests check the adaptation — that the panel gets a usable spec in one
round trip, that a bad spec comes back as a message instead of a 500, that the
JSON crossing the webview boundary is actually serializable, and that both
transports reach the same code.
"""

import json

import pytest

pytest.importorskip("scistackplot")
pytest.importorskip("scistackplotdb")

from scistack_gui.services import plot_service


@pytest.fixture(autouse=True)
def _clear_source_cache():
    plot_service.invalidate()
    yield
    plot_service.invalidate()


# --- describe --------------------------------------------------------------


def test_describe_lists_plottable_variables(populated_db):
    result = plot_service.describe(populated_db)
    names = {m["name"] for m in result["catalog"]["measures"]}

    assert {"RawSignal", "FilteredSignal"} <= names


def test_describe_opens_a_variable_with_a_usable_default_spec(populated_db):
    result = plot_service.describe(populated_db, "RawSignal")

    assert result["eligible"] is True
    assert result["spec"]["measures"] == ["RawSignal"]
    # 1-D data with replicates defaults to a mean line + error band.
    assert result["capabilities"]["shape"] == "1d"
    assert result["spec"]["kind"] in result["capabilities"]["available"]


def test_describe_falls_back_to_the_first_plottable_measure(populated_db):
    """The palette command and the CSV entry point name no variable."""
    result = plot_service.describe(populated_db)
    assert result["variable"] is not None


def test_describe_is_json_serializable(populated_db):
    json.dumps(plot_service.describe(populated_db, "RawSignal"))


def test_unknown_variable_raises_a_key_error(populated_db):
    with pytest.raises(KeyError):
        plot_service.describe(populated_db, "NoSuchVariable")


# --- resolve ---------------------------------------------------------------


def test_resolve_returns_plotly_payloads(populated_db):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    result = plot_service.resolve_figures(populated_db, spec)

    assert result["ok"] is True
    assert result["figures"]
    figure = result["figures"][0]["figure"]
    assert "data" in figure and "layout" in figure
    json.dumps(result)  # must survive the webview boundary


def test_iterate_role_produces_one_payload_per_level(populated_db):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    spec["roles"] = {**spec["roles"], "subject": "iterate"}
    result = plot_service.resolve_figures(populated_db, spec)

    assert len(result["figures"]) == 2  # subjects 1 and 2
    assert {f["label"] for f in result["figures"]} == {"subject=1", "subject=2"}


def test_invalid_spec_returns_a_message_not_an_exception(populated_db):
    """A role conflict is user-correctable state, so the panel shows it."""
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    spec["roles"] = {**spec["roles"], "subject": "x", "session": "x"}
    result = plot_service.resolve_figures(populated_db, spec)

    assert result["ok"] is False
    assert "one factor" in result["error"]
    assert result["figures"] == []


def test_max_points_downsamples_for_transport(populated_db):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    spec["kind"] = "line"
    result = plot_service.resolve_figures(populated_db, spec, max_points=5)

    assert result["figures"][0]["downsampled_from"] is not None


# --- capabilities ----------------------------------------------------------


def test_capabilities_track_role_changes(populated_db):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]

    with_replicates = plot_service.capabilities_for(populated_db, spec)
    assert with_replicates["has_replicates"] is True
    assert "band" in with_replicates["available"]

    spec["roles"] = {key: "color" if key == "session" else "aggregate"
                     for key in spec["roles"]}
    collapsed = plot_service.capabilities_for(populated_db, spec)
    assert collapsed["has_replicates"] is False
    assert "band" not in collapsed["available"]


# --- export ----------------------------------------------------------------


def test_export_generates_a_plot_function_and_call(populated_db):
    pytest.importorskip("seaborn")
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    result = plot_service.export_code(populated_db, spec)

    assert result["function_name"].startswith("plot_")
    assert "for_each(" in result["foreach_source"]
    assert "outputs=[RawSignalFigure]" in result["foreach_source"]
    compile(result["source"], "<generated>", "exec")


def test_export_never_calls_back_into_this_package(populated_db):
    """
    Generated code must run on seaborn alone.

    The docstring does mention scistackplot — the byline, and the embedded
    ``scistackplot-spec:`` block the GUI reads back to repopulate its controls.
    What must never appear is an import or a call, which is what would make an
    exported pipeline depend on this package at runtime.
    """
    pytest.importorskip("seaborn")
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    source = plot_service.export_code(populated_db, spec)["function_source"]

    assert "import scistackplot" not in source
    assert "scistackplot.render" not in source
    assert "sns." in source


def test_add_to_pipeline_writes_the_function_and_declares_the_output(
    client_with_variable_file, populated_db, tmp_path
):
    pytest.importorskip("seaborn")
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    result = plot_service.add_to_pipeline(populated_db, spec)

    assert result.get("error") is None, result.get("error")
    written = tmp_path / "scistack_plots.py"
    assert written.exists()
    assert f"def {result['function_name']}(" in written.read_text()


def test_add_to_pipeline_refuses_to_clobber_an_existing_function(
    client_with_variable_file, populated_db
):
    pytest.importorskip("seaborn")
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    plot_service.add_to_pipeline(populated_db, spec)
    second = plot_service.add_to_pipeline(populated_db, spec)

    assert second["ok"] is False
    assert "already defines" in second["error"]


# --- saving an image -------------------------------------------------------


def test_save_figure_writes_a_png(populated_db, tmp_path):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    target = tmp_path / "figure.png"

    result = plot_service.save_figure(populated_db, spec, str(target))

    assert result["ok"] is True
    assert result["files"] == [str(target)]
    assert target.exists() and target.stat().st_size > 0


def test_save_figure_uses_full_resolution(populated_db, tmp_path):
    """
    The interactive view is downsampled for transport; a saved figure must not
    be. Nothing in save_figure may pass max_points.
    """
    import inspect

    # Drop the docstring, which mentions max_points precisely to explain why the
    # body must not use it.
    body = inspect.getsource(plot_service.save_figure).split('"""', 2)[-1]
    assert "max_points" not in body


def test_save_figure_honours_the_suffix(populated_db, tmp_path):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    target = tmp_path / "figure.svg"

    result = plot_service.save_figure(populated_db, spec, str(target))
    assert result["files"] == [str(target)]
    assert target.exists()


def test_save_figure_writes_one_file_per_iterated_figure(populated_db, tmp_path):
    """A fanned-out spec must not silently save only the first figure."""
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    spec["roles"] = {**spec["roles"], "subject": "iterate"}

    result = plot_service.save_figure(populated_db, spec, str(tmp_path / "emg.png"))

    assert len(result["files"]) == 2
    assert {p.name for p in tmp_path.glob("emg_*.png")} == {
        "emg_subject_1.png",
        "emg_subject_2.png",
    }


def test_save_figure_reports_a_bad_spec_instead_of_raising(populated_db, tmp_path):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    spec["roles"] = {**spec["roles"], "subject": "x", "session": "x"}

    result = plot_service.save_figure(populated_db, spec, str(tmp_path / "x.png"))
    assert result["ok"] is False
    assert result["files"] == []


def test_save_figure_creates_missing_parent_directories(populated_db, tmp_path):
    spec = plot_service.describe(populated_db, "RawSignal")["spec"]
    target = tmp_path / "new" / "nested" / "figure.png"

    assert plot_service.save_figure(populated_db, spec, str(target))["ok"] is True
    assert target.exists()


# --- the CSV path (same protocol, no database) -----------------------------


def test_csv_source_needs_no_database(tmp_path):
    csv = tmp_path / "gait.csv"
    csv.write_text(
        "subject,session,StepLength\n"
        "s01,pre,1.1\ns01,post,1.3\ns02,pre,1.0\ns02,post,1.2\n"
    )

    described = plot_service.describe(None, "StepLength", csv_path=str(csv))
    assert described["eligible"] is True

    resolved = plot_service.resolve_figures(
        None, described["spec"], csv_path=str(csv)
    )
    assert resolved["ok"] is True
    assert resolved["figures"]


def test_csv_padded_ids_keep_numeric_order(tmp_path):
    """Ten subjects: lexicographic order would put s10 second."""
    csv = tmp_path / "wide.csv"
    rows = "\n".join(f"s{n:02d},{n}" for n in range(1, 11))
    csv.write_text(f"subject,Mass\n{rows}\n")

    described = plot_service.describe(None, "Mass", csv_path=str(csv))
    levels = described["table"]["factors"][0]["levels"]
    assert levels == [f"s{n:02d}" for n in range(1, 11)]


# --- both transports reach the same code -----------------------------------


def test_http_route_and_rpc_handler_share_the_service(client, populated_db):
    from scistack_gui.server import METHODS

    response = client.post("/api/plot/describe", json={"variable": "RawSignal"})
    assert response.status_code == 200
    http_result = response.json()

    rpc_result = METHODS["plot_describe"]({"variable": "RawSignal"})
    assert rpc_result["spec"] == http_result["spec"]


# --- cache invalidation after a run ----------------------------------------
#
# ScidbSource caches whole variable frames. A run that writes records makes
# them stale, and a stale frame means the panel plots PRE-RUN data — the
# reported bug, where an edited function's new records were written correctly
# but the figure kept showing the old ones. `plot_invalidate` existed and was
# wired end to end; nothing called it.


def test_source_is_cached_between_calls(populated_db):
    first = plot_service.get_source(populated_db)
    assert plot_service.get_source(populated_db) is first


def test_sources_are_keyed_by_path_not_object_identity(populated_db):
    """`id(db)` was the old key. CPython reuses ids after GC, so a new manager
    could land on a dead entry and inherit another database's frames."""
    plot_service.get_source(populated_db)

    assert ("db", str(populated_db.dataset_db_path)) in plot_service._sources
    assert not any(isinstance(k[1], int) for k in plot_service._sources)


def test_invalidate_drops_the_cached_source(populated_db):
    first = plot_service.get_source(populated_db)
    plot_service.invalidate(populated_db)

    assert plot_service.get_source(populated_db) is not first


def test_invalidate_accepts_a_bare_path(populated_db):
    """The MATLAB run threads have released their connection by the time they
    finish, so they invalidate by path rather than by manager."""
    first = plot_service.get_source(populated_db)
    plot_service.invalidate(populated_db.dataset_db_path)

    assert plot_service.get_source(populated_db) is not first


def test_invalidate_is_a_noop_for_an_unbuilt_source(populated_db):
    """Called after every run, including runs where no panel was ever open."""
    assert plot_service.invalidate(populated_db) == {"ok": True}


def test_run_completion_invalidates_the_cache(populated_db, monkeypatch):
    """The wiring itself: finishing a run must drop the cache, not just push
    dag_updated. These two travelled separately and only one was ever sent."""
    from scistack_gui.api import run as run_api

    cached = plot_service.get_source(populated_db)
    messages = []
    monkeypatch.setattr(run_api, "push_message", messages.append)

    run_api._notify_records_changed()

    assert {"type": "dag_updated"} in messages
    assert plot_service.get_source(populated_db) is not cached


def test_notify_still_updates_the_dag_if_invalidation_fails(
    populated_db, monkeypatch
):
    """A cache we failed to drop is a stale figure, not a failed run — the
    records are already written, so this must never bury the result."""
    from scistack_gui.api import run as run_api

    messages = []
    monkeypatch.setattr(run_api, "push_message", messages.append)
    monkeypatch.setattr(
        plot_service,
        "invalidate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    run_api._notify_records_changed()  # must not raise

    assert {"type": "dag_updated"} in messages


# --- panel styles ----------------------------------------------------------
#
# Source-scanned rather than rendered: the GUI has no frontend test runner, and
# this class of defect is a property of the style OBJECTS, which are readable
# text (same approach as the handle-id guards in test_edge_resolver.py).


def _plot_studio_style(name: str) -> str:
    """The body of one entry in PlotStudio.tsx's ``styles`` map."""
    import re
    from pathlib import Path

    source = (
        Path(__file__).parent.parent
        / "frontend/src/components/PlotStudio/PlotStudio.tsx"
    ).read_text()
    match = re.search(rf"\n  {name}: \{{(.*?)\n?  \}},", source, re.DOTALL)
    assert match, f"PlotStudio.tsx has no {name!r} style"
    return match.group(1)


def test_collapsed_controls_keep_the_scroll_property():
    """
    The controls rail must still scroll after being collapsed and reopened.

    React removes the style properties a re-render drops. When the collapsed
    state added the ``overflow`` SHORTHAND on top of a base that set the
    ``overflowY`` LONGHAND, reopening cleared ``overflow`` — which owns
    overflow-y — while ``overflowY: 'auto'``, unchanged between the two
    objects, was skipped as nothing to re-apply. The rail came back
    unscrollable, and only ever after a collapse/expand cycle.

    So: both states name the same overflow keys, and neither uses the
    shorthand.
    """
    open_style = _plot_studio_style("controls")
    hidden_style = _plot_studio_style("controlsHidden")

    for name, style in (("controls", open_style), ("controlsHidden", hidden_style)):
        assert "overflow:" not in style, (
            f"PlotStudio.tsx styles.{name} uses the `overflow` shorthand; the "
            "two states are merged, so clearing it on reopen also clears the "
            "other state's overflowY. Use overflowX/overflowY."
        )
        assert "overflowY" in style, (
            f"PlotStudio.tsx styles.{name} must state overflowY explicitly — a "
            "property present in only one of the two merged states is the "
            "property React silently drops."
        )

    assert "overflowY: 'auto'" in open_style
