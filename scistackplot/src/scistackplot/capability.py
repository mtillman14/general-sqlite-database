"""
Which plot kinds are available, and which one to pick by default.

This is the single rule behind two of the requirements that look separate:
"different data types get different default plots", and "iterating over a
higher schema level unlocks more summative plot types". Both fall out of one
observation — a distribution needs replicates, and replicates exist only when
some factor is left FREE (not mapped to a channel, not collapsed).

The GUI must render only what ``available_plots`` returns. Plot policy lives
here, not in TypeScript (CLAUDE.md NOTE 3).
"""

from __future__ import annotations

from .shape import Shape
from .spec import PlotKind, PlotSpec, Role
from .table import CODE_FACTOR_PREFIX, LongTable

#: Kinds that summarize several rows per x position into one mark.
DISTRIBUTION_KINDS = (PlotKind.BOX, PlotKind.VIOLIN, PlotKind.BAR, PlotKind.BAND)


def has_replicates(roles: dict[str, Role]) -> bool:
    """
    True when some factor's levels survive as multiple rows per plotted cell.

    AGGREGATE deliberately does not count: it collapses its factor to a mean
    *before* plotting, so it removes replicates rather than providing them.
    Its purpose is noise reduction ("average over trials"), after which a
    remaining FREE factor (e.g. subject) is what supplies the distribution.
    """
    return any(role is Role.FREE for role in roles.values())


def available_plots(
    shape: Shape,
    roles: dict[str, Role],
    *,
    n_measures: int = 1,
) -> list[PlotKind]:
    """Plot kinds that can be rendered for this shape and role assignment."""
    if shape is Shape.MATRIX_2D:
        return [PlotKind.HEATMAP]

    if n_measures >= 2:
        # x comes from a second measure: a relational scatter, optionally with
        # a connecting line when the x measure is ordered.
        return [PlotKind.SCATTER, PlotKind.LINE]

    replicates = has_replicates(roles)

    if shape is Shape.SERIES_1D:
        kinds = [PlotKind.LINE]
        if replicates:
            kinds.append(PlotKind.BAND)
        return kinds

    if shape is Shape.SCALAR:
        kinds = [PlotKind.SCATTER, PlotKind.STRIP]
        if replicates:
            kinds.extend([PlotKind.BOX, PlotKind.VIOLIN, PlotKind.BAR])
        return kinds

    return []


def default_plot(
    shape: Shape,
    roles: dict[str, Role],
    *,
    n_measures: int = 1,
) -> PlotKind | None:
    """
    The kind to select when a table is first opened.

    scalar → scatter, or box once there are replicates to distribute;
    1-D → one line per observation, or a mean line with a shaded error region
    once there are replicates; 2-D → heatmap.
    """
    kinds = available_plots(shape, roles, n_measures=n_measures)
    if not kinds:
        return None

    replicates = has_replicates(roles)
    if n_measures >= 2:
        return PlotKind.SCATTER
    if shape is Shape.SERIES_1D:
        return PlotKind.BAND if replicates else PlotKind.LINE
    if shape is Shape.SCALAR:
        return PlotKind.BOX if replicates else PlotKind.SCATTER
    return kinds[0]


def why_unavailable(kind: PlotKind, shape: Shape, roles: dict[str, Role]) -> str | None:
    """
    Explain a kind's absence, for GUI tooltips on disabled options.

    Returns None when the kind IS available.
    """
    if kind in available_plots(shape, roles):
        return None
    if shape is Shape.MATRIX_2D:
        return "2-D measures render as a heatmap."
    if kind in DISTRIBUTION_KINDS and not has_replicates(roles):
        return (
            "Needs replicates: leave at least one factor 'free' (unassigned) so "
            "each x position has several values to summarize."
        )
    if kind is PlotKind.BAND and shape is not Shape.SERIES_1D:
        return "Error bands apply to 1-D measures."
    if kind is PlotKind.LINE and shape is Shape.SCALAR:
        return "Lines need a 1-D measure or a second measure for the x axis."
    return f"Not available for a {shape} measure."


def capabilities(spec: PlotSpec, table: LongTable) -> dict:
    """
    The full JSON-serializable capability report for the GUI.

    One call gives the panel everything it needs to render its controls:
    which kinds are selectable, why the others are not, and what the default
    would be for the current role assignment.
    """
    from .roles import complete_roles
    from .variants import apply_variant_sets, strip_answered_roles

    # The kinds and roles reported must be the ones the figure will actually be
    # built with, so the derived table (named variants folded into a ``Variant``
    # factor) is what they are computed against — exactly as ``resolve`` does,
    # stale-role drop included, or the panel would offer a role selector for a
    # factor the render is about to reject.
    derived = apply_variant_sets(spec, table)
    spec = strip_answered_roles(spec, table, derived)
    roles = complete_roles(spec, derived)
    shape = derived.shape_of(spec.y_measure)
    n_measures = len(spec.measures)
    allowed = available_plots(shape, roles, n_measures=n_measures)

    return {
        "shape": str(shape),
        "has_replicates": has_replicates(roles),
        "default": str(default_plot(shape, roles, n_measures=n_measures) or ""),
        "available": [str(k) for k in allowed],
        "kinds": [
            {
                "kind": str(kind),
                "available": kind in allowed,
                "reason": why_unavailable(kind, shape, roles),
            }
            for kind in PlotKind
        ],
        "roles": {name: str(role) for name, role in roles.items()},
        "factors": derived.describe()["factors"],
        "variants": variant_summary(spec, table),
    }


def variant_summary(spec: PlotSpec, table: LongTable) -> dict:
    """The variant picker's whole data model, and the combination readout.

    Three things, all measured against the same frame the renderer will use:

    ``sets``
        One entry per named variant — its label (the user's, or the auto one it
        would carry), the selection it holds, and **how many rows it actually
        matched**. That last number is the one that catches real mistakes: a
        variant selecting a combination nobody ever ran is indistinguishable
        from a working one until the series silently fails to appear.
    ``factors``
        Every variant axis in the data with its levels and its ``origin``, so
        the popup can map an axis to the pipeline node that produced it without
        parsing ``Code:`` or ``fn.param`` out of a column name. Taken from the
        table BEFORE selection, deliberately: an axis a set has already pinned
        is precisely the one the user needs to be able to re-open and change.
    ``total_combinations`` / ``selected_combinations``
        How much of the variant space the current selection covers.

    **Why the counts are measured, not computed.** Multiplying level counts
    would be wrong in two ways that matter. The default selection is on
    ``CodeIsLatest``, which is deliberately *not* a variant factor, so a purely
    combinatorial count would report every combination as selected while the
    figure showed half of them. And real data is ragged: a location never re-run
    under the newest code has no row for that combination, so the Cartesian
    product overstates what exists. Both numbers therefore come from the frame,
    through the same :func:`~scistackplot.variants.variant_set_mask` the
    renderer applies — a readout the figure could disagree with would be worse
    than none.

    ``selected_combinations`` of 0 is a legitimate state to display (the user
    has deselected everything); it is ``roles.validate``'s job to refuse
    rendering it, not this function's to hide it.
    """
    import pandas as pd

    from .variants import (
        auto_label,
        defined_sets,
        set_name,
        spanned_code_axes,
        variant_set_mask,
    )

    frame = table.frame
    names = [f.name for f in table.variant_factors if f.name in frame.columns]

    sets = []
    # With nothing selected — no rows, or none filled in yet — every row is on
    # screen, which is what the figure is showing too.
    kept_mask = pd.Series(not defined_sets(spec.variant_sets), index=frame.index)
    claimed = pd.Series(False, index=frame.index)
    for index, variant in enumerate(spec.variant_sets):
        defined = bool(variant.selection)
        if defined:
            mask = variant_set_mask(
                frame, variant.selection, latest_column=table.latest_column
            )
            # First-match-wins, mirroring apply_variant_sets: the count shown
            # must be the number of rows this variant contributes to the figure,
            # not the number it would match on its own.
            fresh = mask & ~claimed
            claimed |= fresh
            kept_mask |= fresh
        else:
            # An unfilled row is inert — it claims nothing and changes nothing,
            # so it must not be reported as having matched nothing either.
            fresh = pd.Series(False, index=frame.index)
        sets.append(
            {
                "name": set_name(variant, index),
                "auto_label": auto_label(variant.selection, index=index),
                "explicit_name": variant.name,
                "selection": dict(variant.selection),
                "defined": defined,
                "row_count": int(fresh.sum()),
                # Code axes this variant leaves open and disagrees on. Reported
                # per row because that is where the fix is (pin a version, or
                # split the row), and because the column itself is no longer
                # offered as a factor.
                "spans": spanned_code_axes(frame[fresh], variant.selection, table)
                if defined
                else {},
            }
        )

    if not names:
        return {
            "sets": sets,
            "factors": [],
            "total_combinations": 0,
            "selected_combinations": 0,
            "policy": str(spec.variant_policy),
        }

    kept = frame[kept_mask]
    as_text = frame[names].astype(str)
    total = len(as_text.drop_duplicates())
    selected = len(kept[names].astype(str).drop_duplicates()) if len(kept) else 0

    factors = []
    for factor in table.variant_factors:
        if factor.name not in frame.columns:
            continue
        levels = [str(level) for level in factor.levels]
        surviving = set(kept[factor.name].astype(str)) if len(kept) else set()
        factors.append(
            {
                "name": factor.name,
                "levels": levels,
                "selected": [level for level in levels if level in surviving],
                # Code axes read differently from experimental conditions —
                # one is usually pinned, the other usually faceted — so the GUI
                # needs to tell them apart without parsing the name itself.
                "is_code": factor.name.startswith(CODE_FACTOR_PREFIX),
                "origin": factor.origin,
            }
        )

    return {
        "sets": sets,
        "factors": factors,
        "total_combinations": total,
        "selected_combinations": selected,
        "policy": str(spec.variant_policy),
        "latest_column": table.latest_column,
    }
