"""Subcube variant pinning, and the picker's data model.

A pin value may be a **list** of levels, not just one. That is what turns "show
me this one variant" into "show me every variant where bandpass=v1" — the
question a project with several versioned layers actually asks, and one that
cannot be expressed a single scalar at a time.

``variant_summary`` is the other half: one row per variant factor plus the
combination counts the GUI reads. Both go through ``variant_pin_mask``, so the
number on screen and the rows in the figure cannot disagree.

See docs/claude/variant-selection.md §5.
"""

from __future__ import annotations

import pandas as pd
from scistackplot import LongTable, PlotSpec, Role, variant_summary
from scistackplot.reduce import variant_pin_mask
from scistackplot.spec import VariantPolicy


def _table(frame: pd.DataFrame, variants: list[str]) -> LongTable:
    factors = [c for c in frame.columns if c != "value"]
    return LongTable.from_frame(
        frame,
        factors=factors,
        measures=["value"],
        variant_factors=variants,
        name="value",
    )


# --- the mask -------------------------------------------------------------


def test_a_level_set_keeps_any_of_them():
    frame = pd.DataFrame({"Code:f": ["v1", "v2", "v3"], "value": [1, 2, 3]})

    kept = frame[variant_pin_mask(frame, {"Code:f": ["v1", "v3"]})]

    assert kept["value"].tolist() == [1, 3]


def test_a_scalar_still_works():
    frame = pd.DataFrame({"Code:f": ["v1", "v2"], "value": [1, 2]})

    kept = frame[variant_pin_mask(frame, {"Code:f": "v2"})]

    assert kept["value"].tolist() == [2]


def test_an_unknown_column_is_ignored_not_matched_against():
    """A spec outlives the table it was written against — a two-measure join
    drops columns, a reload may find a factor gone. A stale key must not
    silently empty the figure."""
    frame = pd.DataFrame({"Code:f": ["v1", "v2"], "value": [1, 2]})

    kept = frame[variant_pin_mask(frame, {"gone": "v9"})]

    assert len(kept) == 2


def test_two_dimensions_select_a_subcube_not_a_point():
    """Neither dimension alone identifies a row; pinning one leaves the other
    free, which is exactly 'every variant where filter=v1'."""
    frame = pd.DataFrame(
        {
            "Code:load": ["v1", "v1", "v2", "v2"],
            "Code:filter": ["v1", "v2", "v1", "v2"],
            "value": [1, 2, 3, 4],
        }
    )

    kept = frame[variant_pin_mask(frame, {"Code:filter": ["v1"]})]

    assert kept["value"].tolist() == [1, 3], "both load versions must survive"


def test_an_empty_level_list_keeps_nothing():
    """A legitimate state to be IN (everything unchecked) even though it is not
    one to render. The picker reports it; validate refuses it."""
    frame = pd.DataFrame({"Code:f": ["v1", "v2"], "value": [1, 2]})

    assert len(frame[variant_pin_mask(frame, {"Code:f": []})]) == 0


# --- the picker's data model ---------------------------------------------


def _two_axis_table() -> LongTable:
    return _table(
        pd.DataFrame(
            {
                "Code:load": ["v1", "v1", "v2", "v2"],
                "Code:filter": ["v1", "v2", "v1", "v2"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        ["Code:load", "Code:filter"],
    )


def test_summary_lists_every_variant_factor_and_its_levels():
    table = _two_axis_table()
    spec = PlotSpec(measures=["value"], roles={}, variant_policy=VariantPolicy.FACET)

    summary = variant_summary(spec, table)

    assert [f["name"] for f in summary["factors"]] == ["Code:load", "Code:filter"]
    assert summary["factors"][0]["levels"] == ["v1", "v2"]
    assert all(f["is_code"] for f in summary["factors"])


def test_unpinned_reports_everything_selected():
    table = _two_axis_table()
    spec = PlotSpec(measures=["value"], roles={}, variant_policy=VariantPolicy.FACET)

    summary = variant_summary(spec, table)

    assert summary["total_combinations"] == 4
    assert summary["selected_combinations"] == 4


def test_a_partial_pin_reports_the_surviving_combinations():
    table = _two_axis_table()
    spec = PlotSpec(
        measures=["value"],
        roles={},
        variant_policy=VariantPolicy.PIN,
        pinned_variant={"Code:filter": ["v1"]},
    )

    summary = variant_summary(spec, table)

    assert summary["selected_combinations"] == 2
    assert summary["total_combinations"] == 4
    by_name = {f["name"]: f for f in summary["factors"]}
    assert by_name["Code:filter"]["selected"] == ["v1"]
    assert by_name["Code:load"]["selected"] == ["v1", "v2"], (
        "the unpinned dimension stays fully selected — that is what makes the "
        "pin partial, and the readout has to show it"
    )


def test_counts_are_measured_not_multiplied():
    """Real data is ragged: a location never re-run under the newest code has
    no row for that combination. Multiplying level counts would claim 4."""
    table = _table(
        pd.DataFrame(
            {
                "Code:load": ["v1", "v1", "v2"],
                "Code:filter": ["v1", "v2", "v1"],
                "value": [1.0, 2.0, 3.0],
            }
        ),
        ["Code:load", "Code:filter"],
    )
    spec = PlotSpec(measures=["value"], roles={}, variant_policy=VariantPolicy.FACET)

    assert variant_summary(spec, table)["total_combinations"] == 3


def test_selecting_nothing_is_reported_not_hidden():
    table = _two_axis_table()
    spec = PlotSpec(
        measures=["value"],
        roles={},
        variant_policy=VariantPolicy.PIN,
        pinned_variant={"Code:filter": []},
    )

    assert variant_summary(spec, table)["selected_combinations"] == 0


def test_a_table_without_variants_summarises_to_nothing():
    table = _table(
        pd.DataFrame({"subject": ["01", "02"], "value": [1.0, 2.0]}), []
    )
    spec = PlotSpec(measures=["value"], roles={"subject": Role.X})

    summary = variant_summary(spec, table)

    assert summary["factors"] == []
    assert summary["total_combinations"] == 0
