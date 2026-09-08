"""
Multi-layer variant identity — what a variable knows about code changes that
happened UPSTREAM of it.

Code versions used to be a **one-hop** discriminator: derived from the immediate
producing invocation only, while ``branch_params`` walked the whole chain. Two
records that differed solely by the version of some *upstream* function reached
the display layer indistinguishable and overplotted as replicates.

``code_versions_batch`` closes that — one ``Code:<fn>`` column per upstream
function holding more than one version. These tests were written first as
``xfail`` characterisations of the bug (2026-09-08) and flipped when Stage 1/2
landed; the "scenario is real" half stays because it is what gives the rest of
the file meaning.

See docs/claude/variant-selection.md §2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scidb import for_each
from scistackplot import PlotKind, PlotSpec, Role, RoleError, validate

from scistackplotdb import ScidbSource, load_variable

from conftest import Scaled, Signal, Summarized


@pytest.fixture
def upstream_code_versions(seeded):
    """Edit an UPSTREAM body, leave the downstream one alone, re-run both.

    The sequence matters and a shorter one does not reproduce the bug. Running
    the second layer only once is not enough: it consumes ``Scaled`` through the
    load path, which collapses to the latest record per variant group, so it
    would see one input and write one output. Two ``Summarized`` records only
    coexist once the whole pipeline is re-run after the edit — which is exactly
    what "the user edited a loader and hit Run" does.

    ``__name__`` is pinned across both bodies so this models a body *edit*
    (``function_name`` stable, ``function_hash`` different) rather than two
    differently-named functions.
    """

    def scale_v1(signal):
        return float(np.mean(signal) * 2)

    def scale_v2(signal):
        return float(np.mean(signal) * 2 + 1)

    def summarize(scaled):
        # NEVER edited. Its hash is identical in both passes, which is the
        # whole point: the only thing distinguishing the two Summarized records
        # is an input produced by code that changed.
        return float(scaled) * 10

    for body in (scale_v1, scale_v2):
        body.__name__ = "scale_signal"
        for_each(
            body,
            inputs={"signal": Signal},
            outputs=[Scaled],
            subject=[],
            session=[],
            trial=[],
        )
        for_each(
            summarize,
            inputs={"scaled": Scaled},
            outputs=[Summarized],
            subject=[],
            session=[],
            trial=[],
        )
    return seeded


# --- the scenario is real -------------------------------------------------


def test_upstream_edit_produces_two_downstream_records(upstream_code_versions):
    """Precondition. Without this the rest of the file proves nothing."""
    loaded = load_variable(upstream_code_versions, "Summarized")

    # 3 subjects x 2 sessions x 2 trials = 12 locations, two records each.
    assert len(loaded.frame) == 2 * 3 * 2 * 2


def test_the_upstream_variable_itself_does_distinguish_them(upstream_code_versions):
    """One hop away it works — this is the 2026-09-06 fix, still holding."""
    loaded = load_variable(upstream_code_versions, "Scaled")

    assert "Code:scale_signal" in loaded.variant_columns
    assert set(loaded.frame["Code:scale_signal"]) == {"v1", "v2"}


def test_downstream_records_share_one_producing_hash(upstream_code_versions):
    """Why the discriminator goes missing: ``summarize`` was never edited, so
    the immediate producing invocation is the same for both records."""
    from scidb.provenance_query import producing_function_versions_batch

    loaded = load_variable(upstream_code_versions, "Summarized")
    versions = producing_function_versions_batch(
        upstream_code_versions._duck, loaded.frame["record_id"].tolist()
    )

    hashes = {info["fn_hash"] for info in versions.values()}
    assert len(hashes) == 1, (
        "the downstream function was not edited, so one hash is correct here — "
        "the bug is that nothing else is consulted"
    )


# --- what we actually want ------------------------------------------------


def test_upstream_code_version_reaches_the_downstream_table(upstream_code_versions):
    """The two Summarized records must be distinguishable by something."""
    loaded = load_variable(upstream_code_versions, "Summarized")

    assert "Code:scale_signal" in loaded.variant_columns, (
        "two records per location differ only by the code that produced their "
        "input; with no variant column they are replicates to every consumer"
    )
    assert set(loaded.frame["Code:scale_signal"]) == {"v1", "v2"}


def test_the_unedited_producer_contributes_no_column(upstream_code_versions):
    """``summarize`` was never edited, so it is not an axis and must not add a
    column whose only level is ``v1``."""
    loaded = load_variable(upstream_code_versions, "Summarized")

    assert "Code:summarize" not in loaded.frame.columns
    assert loaded.variant_columns == ["Code:scale_signal"]


def test_latest_is_chain_wide_not_one_hop(upstream_code_versions):
    """Half the rows are current. Under the one-hop rule every row was, because
    ``summarize``'s own hash never changed — which is what let the stale half
    into the default figure."""
    loaded = load_variable(upstream_code_versions, "Summarized")

    current = loaded.frame[loaded.frame[loaded.latest_column]]
    assert len(current) == 3 * 2 * 2
    assert set(current["Code:scale_signal"]) == {"v2"}


def test_downstream_pooling_is_refused(upstream_code_versions):
    """The guard that exists for exactly this, previously never armed one hop out.

    Mirrors ``test_two_code_versions_are_refused_not_pooled`` in test_source.py
    — same assertion, one layer downstream. The spec deliberately leaves the code
    column unassigned so it defaults to FREE, which is what pooling means here.
    """
    table = ScidbSource(upstream_code_versions).get_table(["Summarized"])
    spec = PlotSpec(
        measures=["Summarized"],
        roles={
            "session": Role.X,
            "subject": Role.FREE,
            "trial": Role.FREE,
        },
        kind=PlotKind.BOX,
    )

    with pytest.raises(RoleError, match="would be pooled"):
        validate(spec, table)
