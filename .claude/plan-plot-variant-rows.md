# Plan — Plot Studio variant rows + variant-selection DAG popup

> Extends `.claude/plan-variant-selection.md` (Stage 5, items 4/5 of its GUI
> list) and `docs/claude/variant-selection.md` §5.
>
> Status: **approved and implemented 2026-09-08. All 8 stages built,
> UNCOMMITTED, Python tests NOT YET RUN (user runs them).** Both vite bundles
> rebuilt.
>
> Two things found during implementation that the plan did not anticipate, both
> fixed, both worth not rediscovering:
>
> 1. **The opening `CodeIsLatest` selection contradicts any explicit version.**
>    Picking `v1` on a node while the flag was still in the selection asks for
>    rows that are simultaneously newest and old — an empty figure, and a
>    control that looks broken. The popup drops the flag on the first explicit
>    choice (`withoutLatestFlag`).
> 2. **The Plot Studio tab mounts no providers.** `PlotRoot` renders the panel
>    directly, so `FunctionNode`'s `useScope`/`useRunLog` and `PipelineNode`'s
>    `usePlanRun` would throw the moment the popup drew them. Both node
>    components now branch at the *component boundary* (hooks cannot be skipped
>    inside one), and the popup substitutes an inert stand-in for
>    `PipelineNode`. This is why the popup mounts only selection state and the
>    canvas only execution state — a stronger guarantee than the plan's.

## What the user asked for

1. Drop the **Measure** section from the top of the rail; move its shape tag to
   the right of the `Plot — {varName}` title.
2. Move **Variants** to the top, above Factors.
3. Give Variants the **Factors shape**: two columns, many rows. Left = an
   editable variant name, auto-populated with a sensible default. Right = a
   button opening a **popup DAG** that mirrors the pipeline canvas, where
   `ParameterNode` checkboxes select the variant's parameter levels and
   `FunctionNode`s gain a **version dropdown** defaulting to `latest`.
4. One variant row by default, with a **`+`** always under the last row.

## Decisions taken with the user (2026-09-08)

- **Each row is a level of a new factor.** The named rows become a synthetic
  `Variant` factor that appears in Factors and takes a role (color / facet /
  separate figures). This is what makes "compare v1 against v3" a figure rather
  than two figures, and it is why the name is worth editing — it labels the
  series. Not a union filter.
- **The rows replace the old variant controls.** The policy `<select>`
  (pin / keep separate / pool) and the per-factor checkbox `VariantPicker` both
  go. One row = today's pin. Several rows = a comparison. `POOL` survives as a
  checkbox on the section, so nothing is lost.
- **The popup draws the whole pipeline** (structure mirrors the canvas), but
  only nodes upstream of the plotted measure get live controls; the rest render
  dimmed and inert with a tooltip. No control that silently does nothing.

## The trap this design exists to avoid

`ParameterNode`'s checkboxes are **execution state** — unchecking one excludes
that value from future `for_each` fan-outs (`pipeline_store.hide_constant_value`).
The popup reuses the *widget* and must never reuse the *scope*: in variant mode
the checkbox writes to `PlotSpec` and issues **no** `hide_parameter_value` /
`unhide_parameter_value` / `set_parameter_group_checked` call. Enforced by
construction (a `VariantSelectionContext`; when it is present the execution
handlers are unreachable) and by a frontend test asserting no backend call fires.

## Data model

A variant row is declarative; the resolution of `"latest"` belongs to the layer
holding the frame, not to TypeScript:

```python
PlotSpec.variant_sets: list[VariantSet]   # VariantSet(name: str | None, selection: dict)
# selection: {"Code:scale_signal": "latest" | "v1" | ["v1","v3"],
#             "scale_signal.factor": ["2", "3"]}
```

- `name=None` means "use the auto label", so a selection edit keeps the label
  honest until the user types over it.
- **`"latest"` resolves in `reduce`, per set**: if *every* code axis in the set
  is `"latest"`, the set filters on `CodeIsLatest` — per schema location, so a
  subject never re-run does not vanish (`load.LATEST_COLUMN`'s whole point).
  If the set pins any specific version, `"latest"` on the remaining axes
  resolves to the **highest ordinal present**, and the GUI shows it as
  `latest (v3)`. Mixing is already in the "named ordinal drops locations that
  never ran it" regime — the user named a version; the honest thing is to say
  which one the others became, not to silently switch semantics.
- **Constrained columns are dropped from the factor list** once the `Variant`
  column is built. Otherwise a two-set spec (v1 and v3) leaves `Code:fn` with
  two levels and `validate` refuses the figure as an unassigned variant — the
  same information encoded twice. Variant columns *no* set constrains stay
  ordinary factors and still must be assigned.
- A row matching **zero rows** is a legitimate state to display (the readout
  says so, loudly); `roles.validate` refuses to render an empty figure.

## Staging

### Stage 1 — `function_versions` (scidb)

`provenance_query.code_version_ordinals` deliberately **omits single-version
functions** ("presence means this is a real axis"). The dropdown needs the
opposite: every version of every function that has run, including the lone one.

- New `function_versions(duck, fn_names)` → `{fn_name: [{"version": "v1",
  "function_hash": …, "first_saved": …}]}`, ordered by earliest save (the
  existing stability guarantee: a new version appends, never renumbers).
- `code_version_ordinals` re-expressed on top of it (`len >= 2` filter), so the
  two can never disagree about what `v2` means.
- Tests: `scidb/tests/test_variant_identity.py` — single-version function
  appears here and not in the ordinals map; ordering stable across a new edit.

### Stage 2 — Axis origin (scistackplotdb)

Today a factor is a bare column name and the GUI would have to parse `"Code:"`
and `"fn.param"` to find the matching node — string-sniffing a scidb
namespacing convention from TypeScript. Instead the structure travels with it.

- `load.attach_variants` also returns `variant_axes`:
  `{"column", "kind": "code"|"param", "function", "param"}` — it already knows
  both (`fn_names` for code columns, `fn.param` keys for branch params).
- `VariableFrame.variant_axes`; `source.get_table` threads them onto
  `FactorInfo.origin`.
- `variant_graph(db, variable)`: axes for the variable + Stage 1 versions for
  every function in its chain. This is the popup's whole data model, built in
  the layer that owns provenance (CLAUDE.md NOTE 3).
- Tests: `scistackplotdb/tests/test_variant_chain.py` — the two-layer fixture's
  axes carry the right function/param, and a single-version function still
  yields a version list.

### Stage 3 — Named variant sets (scistackplot)

- `table.FactorInfo.origin`; `VARIANT_SET_FACTOR = "Variant"`.
- `spec.VariantSet` + `PlotSpec.variant_sets`, in `to_dict`/`from_dict`.
- `reduce`: build the `Variant` column (first matching set wins, unmatched rows
  dropped), drop constrained columns, resolve `"latest"` as above. Logged at
  INFO with rows kept per set — a variant that quietly matched nothing is the
  failure mode worth naming.
- `roles.default_spec`: one set named from `table.default_pin` instead of
  `variant_policy=PIN`; `validate`: the `Variant` factor obeys the same
  unassigned-variant rule, and `PIN` without sets is no longer reachable.
- `capability.variant_summary` grows `sets: [{name, auto_label, row_count,
  selection}]` and puts `origin` on each factor. `auto_label` is composed here
  (it lands in the figure legend, so the plot layer owns it).
- `scistackplotdb.variant_set(name, Variant(...))` is the **scientist-facing
  constructor**: it compiles a `scidb.Variant` into the column-keyed selection
  the spec stores. It lives in scistackplotdb, not scistackplot, so `PlotSpec`
  stays JSON-serializable (docstring round-trip, JSON-RPC) and scistackplot
  stays free of scidb — the CSV source depends on that.
- Tests: `scistackplot/tests/test_variant_pin.py` — set→column mapping, latest
  resolution both ways, constrained-column drop, zero-match set, round-trip.

### Stage 4 — `plot_variant_graph` (GUI backend)

Thin adapter, per `plot_service`'s existing shape: RPC handler in `server.py`,
route in `api/plot.py`, function in `plot_service.py`, delegating to Stage 2.
Test in `scistack-gui/tests/test_plot_service.py`.

### Stage 5 — Sidebar (PlotStudio.tsx)

- `Measure` section deleted; `Shell` gains a `titleTag` rendered right of
  `Plot — {var}`.
- `Variants` section moves above `Factors`: one row per set (name `<input>` +
  `Select on DAG` button), a `+` under the last row, a `pool` checkbox, and the
  combination readout (`4 of 24 combinations`) kept from `VariantPicker`, which
  is otherwise deleted along with the policy `<select>`.
- The `Variant` factor renders in Factors like any other (role `<select>`), fed
  from `capabilities.variants.sets`.

### Stage 6 — The popup DAG

- `PlotStudio/VariantDagPopup.tsx`: a modal (backdrop, Escape/Cancel/Apply)
  over the studio — deliberately *not* a tab, so its different role is obvious.
- Fetches `get_pipeline` + `get_layout` for `main` and applies the same
  `applyDagreLayout`, so the structure and positions are the canvas's, not a
  second layout. Non-draggable, no context menus, no run buttons.
- `VariantSelectionContext` supplies the selection + setter; `ParameterNode`
  and `FunctionNode` branch on it: checkbox → set selection (no RPC),
  `FunctionNode` grows the version `<select>` (`latest`, then `v1…vN` with
  dates from Stage 1). Reusing the components rather than cloning them is what
  keeps "mirrors the canvas exactly" true after the next canvas change.
- Nodes not upstream of the plotted measure (no axis from Stage 2) render
  dimmed with a tooltip.
- Any axis with **no node in this scope** (e.g. produced inside a nested
  pipeline) is listed under the canvas as a plain checkbox row, so an axis is
  never unreachable just because the popup opened at `main`.

### Stage 7 — Bundles and verification

Both vite targets rebuilt (`project_frontend_bundle_rebuild`: a committed .tsx
fix is dead until both are built). Test commands handed over — one package per
invocation (`project_pytest_one_package_at_a_time`):

```
pytest scidb/tests/test_variant_identity.py
pytest scistackplotdb/tests
pytest scistackplot/tests
pytest scistack-gui/tests/test_plot_service.py
```

### Stage 8 — Executable as code (codegen)

Everything the GUI does must be writable by hand — so "export warns about
variants" is not an acceptable end state. Three verified facts set the shape:

- `as_table` frames carry **schema keys + data columns only**
  (`scifor/foreach.py:1507`); variant columns are `attach_variants`' doing, a
  plotting-layer construct. So an endpoint cannot receive one `df` and sort the
  variants out of it — it takes **one input per variant row**.
- An **empty `as_table` frame is valid and does not skip the combo**
  (`scifor/foreach.py:1591`), so a location that only ran `v1` still renders,
  with the absent variant contributing zero rows. Same as the panel.
- **List-valued branch params already mean membership**
  (`database.py:133-146`), so the popup's multi-checkbox selection is
  `low_hz=[20, 50]` with no new scidb work.

The rule tying the two halves together: **a list of variants inside one plot is
one figure with a `Variant` factor; `EachOf` of them is one figure each.** The
same `Variant` object either way — written once, meaning the same thing in a run
and in a figure.

```python
def plot_step_length(baseline, new_filter, filename):
    df = pd.concat([
        baseline.assign(Variant="baseline"),
        new_filter.assign(Variant="new filter"),
    ], ignore_index=True)
    ...

for_each(
    plot_step_length,
    inputs={
        "baseline":   Variant(StepLength, code_version="v1"),
        "new_filter": Variant(StepLength, fn="bandpass", low_hz=[20, 50]),
        "filename":   PathOutput("plots/step_length_{subject}.png"),
    },
    outputs=[StepLengthFigure],
    as_table=["baseline", "new_filter"],
    finalized=True,
    subject=[],
)
```

- `codegen.generate_plot_function` emits the multi-input signature and the
  labeled concat; `endpoint._foreach_call` emits one `Variant(...)` input per
  row, with identifier-safe param names (`"new filter"` → `new_filter`) and the
  display label preserved in the concat.
- Covered by `scistackplotdb/tests/test_fanout_parity.py`, which exists exactly
  to catch preview/export divergence.

## Out of scope

- Re-executing an old version (Stage 4b of the variant plan).
- Node-state correctness for pinned nodes — decided closed, do not re-propose.
- A separate "one figure per variant" control: that is the `Variant` factor in
  the `iterate` role, which the Factors section already offers.
