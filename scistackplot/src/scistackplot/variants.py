"""
Named variants: turning a list of selections into a factor you can plot by.

A :class:`~scistackplot.spec.VariantSet` names a region of variant space —
"baseline" is ``Code:bandpass == v1``, "new filter" is ``low_hz in (20, 50)``.
:func:`apply_variant_sets` collapses however many of those a spec carries into
**one ordinary factor**, ``Variant``, whose levels are the names. From that
point on nothing downstream needs to know variants exist: the factor takes a
role, gets a colour or a facet row, and appears in a legend, exactly like
``session``.

Two decisions here are load-bearing.

**Answered columns leave the factor list.** Once "baseline" *means*
``Code:bandpass == v1``, keeping ``Code:bandpass`` as its own factor states the
same thing twice — and worse, with two sets it is a two-level variant factor
nobody assigned, so ``roles.validate`` would refuse the figure the user just
asked for. The information is not lost; it moved into the name. Which columns
count as answered is :func:`_answered`, and both of its rules matter.

**"latest" resolves against the data, never to a hard-coded ordinal.** See
:func:`resolve_selection` — this is the difference between a figure that keeps
every subject and one that silently drops the subjects nobody re-ran.

See ``docs/claude/variant-selection.md`` and ``.claude/plan-plot-variant-rows.md``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pandas as pd
from scistacklog import Log

from .spec import PlotSpec, VariantSet
from .table import CODE_FACTOR_PREFIX, FactorInfo, LongTable

LAYER = "scistackplot"

#: Column (and factor) the named variants collapse into. Named for what it is
#: to a reader of the figure — the legend says "Variant: baseline", not
#: "Code:bandpass: v1", which is the whole point of letting the user name it.
VARIANT_FACTOR = "Variant"

#: A selection value asking for "whatever is current", rather than a named
#: ordinal. Matches ``scidb.variant.LATEST_VERSION``; duplicated as a literal
#: rather than imported because this package must work with no scidb installed.
LATEST = "latest"

#: Name given to the variant a table opens on — the source's own "these rows are
#: the current ones" recommendation. Named rather than auto-labelled because
#: ``CodeIsLatest=True`` is a flag, not a coordinate, and "CodeIsLatest=True" in
#: a legend tells a reader nothing.
CURRENT_VARIANT_NAME = "current"

_ORDINAL = re.compile(r"^v(\d+)$")


def is_code_axis(column: str) -> bool:
    return column.startswith(CODE_FACTOR_PREFIX)


def resolve_selection(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    *,
    latest_column: str | None = None,
) -> dict[str, Any]:
    """
    Replace every ``"latest"`` in ``selection`` with something the frame can test.

    Two resolutions, and which one applies depends on the rest of the set:

    * **Nothing else pinned** — every code axis in the selection says ``latest``.
      The set filters on ``latest_column``, the per-schema-location flag, and the
      code axes drop out of the selection entirely. This is the important case
      and the reason ``latest`` is not simply "the highest ordinal": a subject
      never re-run under the newest code keeps contributing its own newest
      record instead of vanishing from the figure.
    * **Something else is pinned** to a named ordinal. The location-wise flag is
      no longer usable — a record pinned to ``v1`` is by definition not the
      latest — so the remaining ``latest`` axes resolve to the **highest ordinal
      present in the data**. That does drop locations which never ran it, but
      the user already asked for that by naming a version; the honest thing is
      to say which ordinal it became (the GUI shows ``latest (v3)``), not to
      quietly switch semantics.

    Keys naming a column the frame does not have are dropped, not treated as
    matching nothing: a spec outlives the table it was written against, and a
    stale key must never silently empty a figure.
    """
    present = {k: v for k, v in selection.items() if k in frame.columns}
    dropped = [k for k in selection if k not in frame.columns]
    if dropped:
        Log.debug(
            "variant selection ignores %s — not column(s) of this table",
            dropped,
            layer=LAYER,
        )

    latest_axes = [k for k, v in present.items() if _is_latest(v)]
    if not latest_axes:
        return present

    pinned_elsewhere = any(k not in latest_axes for k in present)
    resolved = {k: v for k, v in present.items() if k not in latest_axes}

    if not pinned_elsewhere and latest_column and latest_column in frame.columns:
        resolved[latest_column] = True
        return resolved

    for axis in latest_axes:
        highest = _highest_ordinal(frame[axis])
        if highest is not None:
            resolved[axis] = highest
    return resolved


def _is_latest(value: Any) -> bool:
    return isinstance(value, str) and value == LATEST


def _highest_ordinal(column: pd.Series) -> Any:
    """The largest ``vN`` in a code column, or None when it holds none.

    Sorted numerically on the ordinal rather than lexically, so ``v10`` beats
    ``v9``. Non-ordinal levels (scidb's ``(n/a)`` for a record whose chain never
    ran this function) are ignored rather than compared against.
    """
    ordinals = []
    for value in column.dropna().astype(str).unique():
        match = _ORDINAL.match(value)
        if match:
            ordinals.append((int(match.group(1)), value))
    if not ordinals:
        return None
    return max(ordinals)[1]


def variant_set_mask(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    *,
    latest_column: str | None = None,
) -> pd.Series:
    """Rows this selection keeps — the single definition of what a variant selects.

    Public because the GUI reports *"4 of 24 combinations"* and a per-row count
    per variant, and both have to be measured with exactly the rule the renderer
    will apply. A second implementation that counted differently would put a
    number on screen the figure disagrees with, which is worse than no number.

    A list/tuple/set value means "any of these" — the subcube rule the popup's
    checkboxes produce, and the same membership semantics
    ``scidb.database._match_branch_param`` already gives a list-valued
    ``Variant(...)`` kwarg.
    """
    mask = pd.Series(True, index=frame.index)
    resolved = resolve_selection(frame, selection, latest_column=latest_column)
    for key, value in resolved.items():
        column = frame[key]
        if isinstance(value, bool):
            # The latest flag is a real bool column; comparing it as text would
            # depend on how pandas spells True.
            mask &= column.fillna(False).astype(bool) == value
            continue
        as_text = column.astype(str)
        if isinstance(value, (list, tuple, set, frozenset)):
            mask &= as_text.isin({str(v) for v in value})
        else:
            mask &= as_text == str(value)
    return mask


def auto_label(selection: dict[str, Any], *, index: int = 0) -> str:
    """The name a variant gets until the user types over it.

    Built from the selection so it stays true while the row is being edited:
    ``bandpass v1 · low_hz=20``. The label has to survive being read in a legend
    a week later, where "Variant 1" would say nothing at all.

    An empty selection is ``(not set)``, not "all variants": an unfilled row is
    inert (:func:`defined_sets`), so a label promising every variant would
    describe a row that contributes nothing.
    """
    parts: list[str] = []
    for column, value in selection.items():
        text = _level_text(value)
        if is_code_axis(column):
            parts.append(f"{column[len(CODE_FACTOR_PREFIX):]} {text}")
        else:
            # Branch params arrive namespaced (``bandpass.low_hz``); the
            # function is usually obvious from context in a legend, the
            # parameter never is.
            parts.append(f"{column.rsplit('.', 1)[-1]}={text}")
    if not parts:
        return "(not set)" if index == 0 else f"(not set {index + 1})"
    return " · ".join(parts)


def _level_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set, frozenset)):
        return "+".join(str(v) for v in value)
    return str(value)


def set_name(variant_set: VariantSet, index: int) -> str:
    return variant_set.name or auto_label(variant_set.selection, index=index)


def defined_sets(sets: list[VariantSet]) -> list[VariantSet]:
    """The variants that actually select something.

    A row with an empty selection is a row the user has added but not filled in
    yet, and it is **inert**: it claims no data, contributes no level, and
    decides nothing about which columns the Variants section answers. Treating
    it as "all variants" instead — which is what an empty selection means once
    applied — made adding a row change the figure before the user had said
    anything about it, and un-answered the code axis for every *other* row,
    dropping `Code:<fn>` back into Factors with a pooling error attached.

    Clicking "+" is not a statement about the data. Nothing should happen until
    the row says something.
    """
    return [variant for variant in sets if variant.selection]


def _answered(table: LongTable, sets: list[VariantSet]) -> set[str]:
    """Columns the named variants have already accounted for.

    These leave the factor list: once "baseline" *means* ``Code:bandpass == v1``,
    keeping the column as its own factor states the same thing twice, and with
    two variants it is an unassigned two-level variant factor that
    ``roles.validate`` refuses — rejecting the very comparison the user asked
    for.

    Two rules, and they differ by axis kind on purpose.

    **Code axes belong to the Variants section, entirely.** Once any variant is
    defined, every ``Code:<fn>`` column is answered — whether that variant named
    a version, asked for ``latest``, or selected on the chain-wide
    ``CodeIsLatest`` flag. "Which version of the code" is the question the
    variant rows exist to answer, so offering the same question again as a
    factor is asking the user to decide the same thing twice, in two places,
    with no way to tell which one wins.

    That deliberately allows a variant to hold rows built by different ordinals.
    Usually that is exactly right — under ``latest``, subject A on v2 and
    subject B on v1 are each the newest AT THEIR OWN LOCATION, which is what
    "current" means. Where it is *not* obviously right, the row says so rather
    than the column coming back: see :func:`spanned_code_axes`.

    **Branch-param axes are answered only when EVERY variant answers them** — an
    intersection, not a union. Nothing about "current code" decides which filter
    cutoff to plot, so if one variant pins ``low_hz == 20`` while another leaves
    it open, the second still holds both cutoffs and the user must still say
    what to do with them.
    """
    sets = defined_sets(sets)
    if not sets:
        return set()
    frame = table.frame
    answered = {column for column in frame.columns if is_code_axis(column)}
    per_variant: list[set[str]] = []
    for variant in sets:
        per_variant.append(
            set(
                resolve_selection(
                    frame, variant.selection, latest_column=table.latest_column
                )
            )
        )
    return answered | set.intersection(*per_variant)


def spanned_code_axes(
    rows: pd.DataFrame, selection: dict[str, Any], table: LongTable
) -> dict[str, int]:
    """Code axes this variant left open and that its rows disagree on.

    The honesty mechanism that lets code axes leave the factor list
    unconditionally. A variant which neither names a version nor asks for the
    current one, but whose rows were built by two different versions of a
    function, is pooling code versions — and the figure will look exactly like
    one that is not. Rather than resurrecting the column as a factor (asking the
    user to answer in Factors a question they are already answering in
    Variants), the *row* reports it.

    A selection resolving through the latest flag is never counted: spanning
    ordinals is what per-location "latest" means, and warning about it would cry
    wolf on the most ordinary state there is.
    """
    resolved = resolve_selection(
        rows, selection, latest_column=table.latest_column
    )
    if table.latest_column and table.latest_column in resolved:
        return {}
    spans: dict[str, int] = {}
    for column in rows.columns:
        if not is_code_axis(column) or column in resolved:
            continue
        levels = rows[column].dropna().astype(str).nunique()
        if levels > 1:
            spans[column] = int(levels)
    return spans


def strip_answered_roles(
    spec: PlotSpec, table: LongTable, derived: LongTable
) -> PlotSpec:
    """Drop roles naming factors the variants answered.

    A role assigned before a variant claimed its column is not a mistake, it is
    stale — and ``validate`` would call it an unknown factor and refuse to draw
    anything. Two ways in, both ordinary:

    * ``default_roles`` puts a multi-level ``Code:<fn>`` on COLOUR, and the
      table then opens on the "current" variant, which answers it;
    * the user assigns a code axis to a facet, then adds a variant that pins it.

    Only names that WERE factors of the undecided table are dropped. A role
    naming something that was never a factor at all is still a typo, and
    ``validate`` should still say so.
    """
    stale = [
        name
        for name in spec.roles
        if table.has_factor(name) and not derived.has_factor(name)
    ]
    if not stale:
        return spec
    Log.debug(
        "dropping role(s) %s — the named variants now account for those columns",
        stale,
        layer=LAYER,
    )
    return replace(
        spec, roles={k: v for k, v in spec.roles.items() if k not in stale}
    )


def apply_variant_sets(spec: PlotSpec, table: LongTable) -> LongTable:
    """
    Fold ``spec.variant_sets`` into a ``Variant`` factor on a derived table.

    Returns ``table`` unchanged when the spec names no variants — or none that
    say anything yet (:func:`defined_sets`) — so a project that never edited a
    function or swept a parameter pays nothing and behaves exactly as before,
    and a half-added row changes nothing until it is filled in.

    Rows matching no set are dropped: the sets are the figure's subject, and a
    row belonging to none of them was not asked for. A row matching several
    goes to the **first** — overlapping selections are legal (``all where
    low_hz=20`` and ``all where code=v1`` genuinely intersect) and duplicating
    the row into both would double-count it in every mean.
    """
    sets = defined_sets(spec.variant_sets)
    if not sets:
        return table

    frame = table.frame
    names = [set_name(s, i) for i, s in enumerate(sets)]
    assigned = pd.Series(pd.NA, index=frame.index, dtype="object")
    counts: list[int] = []

    for name, variant in zip(names, sets, strict=True):
        mask = variant_set_mask(
            frame, variant.selection, latest_column=table.latest_column
        )
        fresh = mask & assigned.isna()
        counts.append(int(fresh.sum()))
        assigned[fresh] = name

        # Code axes leave the factor list unconditionally, so a variant that
        # quietly straddles two versions has to say so here — this is the log
        # half of what the GUI shows on the row.
        spans = spanned_code_axes(frame[fresh], variant.selection, table)
        if spans:
            Log.warn(
                "variant %r pools %s — its rows were built by more than one "
                "version of that code. Pin a version on it, or split it into "
                "one variant per version.",
                name,
                ", ".join(f"{column} ({n} versions)" for column, n in spans.items()),
                layer=LAYER,
            )

    kept = frame[assigned.notna()].copy()
    kept[VARIANT_FACTOR] = assigned[assigned.notna()]

    factors = [f for f in table.factors if f.name not in _answered(table, sets)]
    factors.insert(
        0,
        FactorInfo(
            name=VARIANT_FACTOR,
            # Declared order, not the order the data happens to be in: the rows
            # are a list the user arranged, and a legend that reorders itself
            # when a variant loses its last record is disorienting.
            levels=[name for name, count in zip(names, counts, strict=True) if count],
            is_variant=True,
        ),
    )

    empty = [name for name, count in zip(names, counts, strict=True) if not count]
    if empty:
        # Never silent: a variant that matched nothing is either a typo or a
        # pipeline that was never run, and both look identical to "it worked"
        # if the only symptom is a missing series.
        Log.warn(
            "variant(s) %s matched no rows — they contribute nothing to the "
            "figure. Check the selection against what has actually run.",
            empty,
            layer=LAYER,
        )
    Log.info(
        "variant sets kept %d of %d row(s): %s",
        len(kept),
        len(frame),
        dict(zip(names, counts, strict=True)),
        layer=LAYER,
    )

    return replace(table, frame=kept, factors=factors)
