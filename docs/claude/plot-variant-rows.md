# Named Variant Rows, and the Variant-Selection DAG

> Status: **built 2026-09-08, uncommitted, Python tests not yet run.**
> Plan: `.claude/plan-plot-variant-rows.md`. Prerequisite reading:
> `variant-selection.md` (what a variant *is*, and why `Code:<fn>` columns
> exist), `plotting-library-design.md` (roles, the `PlotSpec`).
>
> This is the GUI half of `variant-selection.md` §5, plus the one thing that
> document left open: how a variant figure is written **as code**. Four things
> here are non-obvious enough to be worth reading before touching any of it.

## 1. What a variant row is

The Plot Studio's Variants section holds **named rows**. One row is one
`scistackplot.VariantSet` — a label and a selection:

```python
VariantSet(name="baseline", selection={"Code:bandpass": "v1"})
VariantSet(name=None,       selection={"bandpass.low_hz": ["20", "50"]})
```

- **One row is a pin.** The figure shows that variant and nothing else. This is
  what a table opens on (`default_spec` seeds one row named `current` from the
  source's `default_pin`), which is why the section always has something to
  edit rather than starting empty.
- **Two or more rows are a comparison.** They collapse into a synthetic
  `Variant` factor whose levels are the names, and it takes a role — colour,
  facet, separate figures — like any other factor. That is the whole feature:
  "v1 against v3" becomes one figure instead of two.
- **`name=None` means "call me whatever my selection says"** (`auto_label`:
  `bandpass v1 · low_hz=20`). The GUI keeps it None until the user types over
  it, so the label stays true while the selection is still being edited.

`VariantPolicy.PIN` and `PlotSpec.pinned_variant` **no longer exist**. Pinning
is what a one-row `variant_sets` *is*; a policy meaning "obey the rows" beside
rows that already say what to keep was two switches for one decision, and the
state where the policy said `facet` while a row said `v1` had no defensible
meaning. `POOL` survives, as a checkbox, answering the only question left: what
happens to variant factors *nothing* selected.

### Selection keys are frame columns, not scidb objects

`selection` is keyed by column (`"Code:bandpass"`, `"bandpass.low_hz"`), never
by a `scidb.Variant`. Two reasons, both hard constraints:

- a spec round-trips through JSON-RPC **and** through a generated docstring
  (`codegen.extract_spec`), so it has to be plain data;
- `scistackplot` must keep working with **no scidb installed** — the CSV source
  depends on that.

Translating a `scidb.Variant` into these keys is therefore
`scistackplotdb.variant_set`, in the layer that knows both vocabularies.

## 2. The two resolutions of "latest" — the subtlest thing here

A code axis may be selected as `"latest"` rather than a named ordinal, and it
resolves **against the data**, in one of two ways
(`scistackplot.variants.resolve_selection`):

| situation | resolves to | why |
|---|---|---|
| every code axis in the row says `latest` | the per-row `CodeIsLatest` flag | Per **schema location**. A subject nobody re-ran keeps contributing its own newest record instead of vanishing from the figure. |
| the row also pins a named ordinal somewhere | the **highest ordinal present** on the remaining axes | The flag is unusable once something is pinned to old code — a `v1` record is by definition not the latest. |

The second case *does* drop locations that never ran that ordinal. That is
acceptable only because the user already asked for it by naming a version, and
it is never silent: the GUI shows the substitution as `latest (v3)`.

Getting the first case wrong is how subjects disappear from a figure with no
error anywhere. `"latest"` is **not** a synonym for "the highest ordinal", here
or in `scidb.Variant(code_version="latest")` — same rule, same reason.

## 3. What leaves the factor list, and what an unfilled row does

Once "baseline" *means* `Code:bandpass == v1`, `apply_variant_sets` removes
`Code:bandpass` from the factor list. This is not tidying:

- keeping it states the same thing twice, and
- with two rows the leftover column has two levels and no role, so
  `roles.validate` refuses the figure — **rejecting exactly the comparison the
  user just asked for.**

`_answered` decides this, and the rule differs by axis kind:

- **Code axes belong to the Variants section, entirely.** Once *any* variant is
  defined, every `Code:<fn>` column is answered — whether the variant named a
  version, asked for `latest`, or selected the chain-wide `CodeIsLatest` flag.
  "Which version of the code" is the question the rows exist to answer, and
  offering it again in Factors asks the user to decide the same thing twice in
  two places with no way to know which wins.
- **Branch params are answered only when *every* variant answers them** — an
  intersection. Nothing about "current code" decides which filter cutoff to
  plot, so a variant that leaves `low_hz` open still owes the user a decision.

Because code axes leave unconditionally, a variant that neither pins a version
nor asks for the current one, but whose rows were built by two versions, is
pooling code silently. `spanned_code_axes` catches that and reports it **on the
row** (`spans` in `variant_summary`, an amber "pools 2 versions" tag, and a
`Log.warn`) — the fix is on the row (pin it, or split it), not in Factors. A
selection resolving through the latest flag is never counted: spanning ordinals
across locations is what per-location "latest" *means*.

### An unfilled row is inert

A row with an **empty selection** — what "+ Add variant" creates — is skipped
everywhere: `defined_sets` filters it out of `apply_variant_sets`, `_answered`,
and `codegen`. It claims no rows, contributes no level, and decides nothing.

This is load-bearing, not politeness. Treating an empty selection as "all
variants" (which is what it means once applied) meant clicking "+" changed the
figure before the user had said anything, *and* un-answered the code axis for
every other row — dropping `Code:<fn>` back into Factors with a pooling error
attached. Clicking "+" is not a statement about the data.

The row still appears in the sidebar, labelled `(not set)` with a grey tag
(grey, not amber: "not yet said" is a state, not a problem — the amber "no data"
tag means a *defined* selection matched nothing, which is a real warning). If
every row is unfilled, that is the same as having none: the code axis returns to
Factors and the pooling guard fires, because now nothing has been selected at
all.

Rows matching no selection are dropped; a row matching several goes to the
**first** (overlapping selections are legal, and duplicating a row would
double-count it in every mean).

### Stale roles

Answering a column can strand a role that named it — `default_roles` puts a
multi-level `Code:<fn>` on colour, and the opening "current" variant then
answers it; or the user facets by a code axis and then pins it. `validate` would
call that an *unknown factor* and refuse to draw anything. Two defences:
`default_spec` derives its roles from the **resolved** table so the bad role is
never created, and `strip_answered_roles` drops one arriving from a saved spec.
Only names that *were* real factors are dropped — a typo still errors.

One related default: a multi-level `Variant` factor with no explicit role gets
**COLOR**, not FREE (`complete_roles`). FREE is not the conservative choice
there — it overplots two variants the user has just gone to the trouble of
naming, and `validate` would reject it a moment later anyway, so the honest
alternatives are "colour it" or "show an error instead of the figure".

## 4. Writing the same figure as code

Everything the GUI does must be writable by hand, so a variant figure has to
survive export. Three verified facts set its shape:

- **`as_table` frames carry schema keys and data columns only**
  (`scifor/foreach.py:1507`). Branch-param and `Code:<fn>` columns are
  `scistackplotdb.attach_variants`' doing — a *plotting-layer* construct that
  never reaches an endpoint. So an endpoint **cannot** receive one `df` and sort
  the variants out of it.
- **An empty `as_table` frame is valid and does not skip the combo**
  (`scifor/foreach.py:1591`), so a location that only ran `v1` still renders,
  with the absent variant contributing zero rows — the same behaviour as the
  panel.
- **List-valued branch params already mean membership**
  (`scidb/database.py:133-146`), so the popup's multi-checkbox selection is
  `low_hz=["20", "50"]` with no new scidb work.

Hence **one input per variant**, labelled and stacked inside the generated
function:

```python
def plot_step_length(baseline, new_filter, filename):
    df = pd.concat([
        baseline.assign(**{"Variant": "baseline"}),
        new_filter.assign(**{"Variant": "new filter"}),
    ], ignore_index=True)
    ...

for_each(
    plot_step_length,
    inputs={
        "baseline":   Variant(StepLength, fn="bandpass", code_version="v1"),
        "new_filter": Variant(StepLength, fn="bandpass", low_hz=["20", "50"]),
        "filename":   PathOutput("plots/step_length_{subject}.png"),
    },
    outputs=[StepLengthFigure],
    as_table=["baseline", "new_filter"],
    finalized=True,
    subject=[],
)
```

The rule tying the interactive and pipeline halves together:

> **A list of variants inside one plot is one figure with a `Variant` factor;
> `EachOf` of them is one figure each.**

Same `Variant` object either way — written once, meaning the same thing in a run
and in a figure. There is deliberately no second control for "one figure per
variant": that is the `Variant` factor in the `iterate` role.

Mechanics worth knowing:

- `codegen.variant_params` makes identifier-safe parameter names from labels
  (`"20 Hz + latest"` → `v_20_hz_latest`) and keeps the display label for the
  `assign`.
- `endpoint.variant_expression` is the inverse of `variants.selection_for`. A
  selection spanning two producing functions **nests** —
  `Variant(Variant(X, fn="loadEMG", code_version="v1"), fn="bandpass", low_hz="20")`
  — because one `Variant(fn=…)` cannot cover two functions, and
  `**{"__code__.loadEMG": "v1"}` would leak a reserved namespace into code the
  user is meant to edit.
- The `CodeIsLatest` selection key becomes `code_version="latest"`: scidb spells
  the same per-location rule the same way.
- `generate_script` (the standalone CSV path) splits the one frame with literal
  pandas instead, and says in a comment when a `"latest"` selection cannot be
  honoured — a flat table carries no provenance to resolve it against.

## 5. The popup: same widgets, opposite meaning

`VariantDagPopup` draws the pipeline canvas in a modal. Parameter nodes offer
their recorded levels as checkboxes; function nodes offer their recorded
versions as a dropdown defaulting to `latest`.

**The trap it is built around.** On the canvas, a `ParameterNode` checkbox is
*execution* state — unchecking a value excludes it from future `for_each`
fan-outs (`pipeline_store.hide_constant_value`). In the popup the identical
widget is *display* state. Binding one to the other would make looking at a plot
quietly rewrite the run configuration.

Three things keep them apart, and the third is stronger than the design asked
for:

1. `VariantSelectionContext` carries the mode. Present → selection; absent (the
   canvas, always) → nothing about existing behaviour changes.
2. A test asserts the variant-mode components never reference `callBackend`
   (`scistack-gui/tests/test_plot_service.py`).
3. **The branch is at the component boundary, not inside a component body** —
   `FunctionNode` returns either `VariantFunctionNode` or
   `PipelineFunctionNode`. This was forced by a crash, not chosen for elegance:
   `PlotRoot` mounts the panel with **no providers**, so `useScope`,
   `useRunLog` and `usePlanRun` throw the instant the popup draws a function or
   pipeline node, and hooks cannot be skipped inside a component. The upshot is
   that the popup mounts *only* selection state and the canvas *only* execution
   state.

`PipelineNode` is replaced outright by an inert stand-in for the same reason;
its insides are a different scope anyway.

### Why the whole pipeline, not the relevant subgraph

`variant-selection.md` §5 proposed the induced subgraph (ancestors contributing
more than one level). The whole canvas won because it is a graph the user
already knows, and because it makes the *absence* of a control informative: a
dimmed node is saying "I do not distinguish these records". Nodes with no axis
render inert with a tooltip rather than offering controls that would do nothing.

### The default selection contradicts an explicit version

A row opens on `{CodeIsLatest: true}`. Adding `Code:f == v1` on top asks for rows
that are simultaneously the newest and the old version — an empty figure, from a
control that looks broken. The popup drops the flag on the first explicit choice
(`withoutLatestFlag`); an explicit choice supersedes the shortcut, which is also
what the user means by making it.

### Axes with no node on this canvas

The popup opens at the root scope, so an axis produced inside a nested pipeline
has no node to click. Those are listed as plain checkbox rows beneath the graph
— an axis is never unreachable just because of where the popup opened.

## 6. Where each piece lives

| concern | home | why there |
|---|---|---|
| every recorded version of a function | `scidb.provenance_query.function_versions` | Provenance. `code_version_ordinals` is now expressed in terms of it, so the two cannot disagree about what `v2` means. |
| which axes a variable has, and their origin | `scistackplotdb.load.attach_variants` → `VariableFrame.variant_axes` → `FactorInfo.origin` | Both halves (`Code:<fn>`, `fn.param`) are in hand there. A GUI splitting those strings would re-implement scidb's namespacing one layer away, and break first when it changes. |
| axes + versions for the popup | `ScidbSource.variant_graph` | Reuses the source's frame cache; opening a dialog must not re-read the variable. |
| `scidb.Variant` → selection | `scistackplotdb.variant_set` | Needs both vocabularies; scistackplot must stay scidb-free. |
| what a selection keeps | `scistackplot.variants.variant_set_mask` | One definition, so the GUI's "4 of 24" and the renderer's rows cannot disagree. |
| the rows' data model + counts | `scistackplot.capability.variant_summary` | Adds `sets` (label, auto label, **row_count**) beside the axes. `row_count == 0` is the number that catches real mistakes. |

## 7. Known limits

- **The popup is root-scope only** (§5).
- **`FactorInfo.levels` are computed on the unfiltered frame** and are not
  recomputed after a selection, so a legend may carry a level with no rows.
  Pre-existing behaviour, kept deliberately: it makes level order stable as
  boxes are toggled.
- **MATLAB source is still not captured** (`variant-selection.md` §3), so
  MATLAB versions remain selectable but not inspectable.
- **Re-running an old version is still not possible** — Stage 4b of
  `.claude/plan-variant-selection.md`. Everything here selects among records
  that already exist.
