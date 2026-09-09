/**
 * Plot Studio — interactive plotting for a scidb variable.
 *
 * The panel is deliberately thin. It renders whatever `plot_describe` and
 * `plot_capabilities` return and sends back a spec; every decision about which
 * plot kinds are legal, what the defaults are, and how data is reduced lives in
 * scistackplot (CLAUDE.md NOTE 3). Adding a plot kind or a role should require
 * no change here beyond a label.
 *
 * The controls replace the four mutually-dependent checkbox groups of the
 * original R/Shiny app with one rule: every factor carries exactly one role.
 * That is a single <select> per factor, and the invariant is enforced by the
 * backend rather than by widgets updating each other's options.
 *
 * See docs/claude/plotting-library-design.md.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-cartesian-dist-min'
import { callBackend, isVSCodeMode } from '../../api'
import VariantDagPopup from './VariantDagPopup'

const Plot = createPlotlyComponent(Plotly)

type Role = 'iterate' | 'x' | 'color' | 'facet' | 'aggregate' | 'free'

const ROLE_OPTIONS: { value: Role; label: string; hint: string }[] = [
  { value: 'x', label: 'X axis', hint: 'Position along the x axis' },
  { value: 'color', label: 'Color', hint: 'One coloured series per level' },
  { value: 'facet', label: 'Facet', hint: 'One subplot per level — arrange them under Layout' },
  { value: 'iterate', label: 'Separate figures', hint: 'One whole figure per level' },
  { value: 'aggregate', label: 'Average over', hint: 'Collapse this factor to its mean' },
  { value: 'free', label: 'Replicates', hint: 'Keep as repeated observations' },
]

const KIND_LABELS: Record<string, string> = {
  scatter: 'Scatter',
  strip: 'Strip (jittered)',
  line: 'Lines',
  box: 'Box',
  violin: 'Violin',
  bar: 'Bar + error',
  band: 'Mean + error band',
  heatmap: 'Heatmap',
}

interface FactorInfo {
  name: string
  display: string
  levels: (string | number)[]
  level_count: number
  is_variant: boolean
  /** Levels are the measure's own struct/dict fields, not a condition. */
  is_field: boolean
}

interface TableInfo {
  factors: FactorInfo[]
  measures: { name: string; shape: string }[]
  row_count: number
}

interface KindInfo {
  kind: string
  available: boolean
  reason: string | null
}

/** One variant factor as the picker renders it. Built entirely by
 *  `capability.variant_summary` — the GUI decides nothing about variants. */
interface VariantFactorInfo {
  name: string
  levels: string[]
  /** Levels surviving the current selection, measured against the frame. */
  selected: string[]
  /** A code-version axis (`Code:bandpass`) rather than an experimental one. */
  is_code: boolean
}

/** One row of the Variants section, as the backend reports it back. */
interface VariantSetInfo {
  /** What to show: the user's name, or the auto label when they haven't typed. */
  name: string
  auto_label: string
  /** null until the user names it — that is what keeps the label following the
   *  selection while it is still being edited. */
  explicit_name: string | null
  selection: Record<string, unknown>
  /** False for a row added but not filled in yet. Such a row is inert — it
   *  changes nothing about the figure until it says something. */
  defined: boolean
  /** Rows this variant contributes to the figure. Zero is the number worth
   *  showing: a variant selecting a combination nobody ran looks exactly like a
   *  working one until its series fails to appear. */
  row_count: number
  /** Code axes this variant left open and whose versions its rows disagree on,
   *  `{column: version count}`. Reported here because the code columns are no
   *  longer offered as factors — the fix is on this row, not in Factors. */
  spans: Record<string, number>
}

interface VariantSummary {
  sets: VariantSetInfo[]
  factors: VariantFactorInfo[]
  /** Combinations present in the data — measured, not level counts multiplied,
   *  because real data is ragged and the default selection is on a non-factor
   *  flag. */
  total_combinations: number
  selected_combinations: number
  policy: string
}

interface Capabilities {
  shape: string
  has_replicates: boolean
  default: string
  available: string[]
  kinds: KindInfo[]
  /** Factors of the RESOLVED table — the synthetic `Variant` factor included,
   *  and the columns its selections consumed excluded. The panel renders these
   *  rather than `describe.table.factors`, which is the pre-selection view. */
  factors?: FactorInfo[]
  variants?: VariantSummary
}

type MatchOp = 'starts_with' | 'ends_with' | 'contains' | 'not_contains' | 'equals' | 'regex'

const MATCH_OPS: { value: MatchOp; label: string }[] = [
  { value: 'starts_with', label: 'starts with' },
  { value: 'ends_with', label: 'ends with' },
  { value: 'contains', label: 'contains' },
  { value: 'not_contains', label: 'does not contain' },
  { value: 'equals', label: 'is exactly' },
  { value: 'regex', label: 'matches regex' },
]

interface Matcher {
  op: MatchOp
  value: string
  label?: string | null
}

interface FacetOptions {
  /** Grid size. null/absent means "compute me from the other one and the panel count". */
  n_rows?: number | null
  n_cols?: number | null
  /** One entry per row / column slot; a blank value is an unset slot. */
  rows?: Matcher[]
  cols?: Matcher[]
  share_x?: boolean
  share_y?: boolean
}

/**
 * What the backend actually laid out, echoed back on every figure.
 *
 * The panel never computes a grid itself: `rows`/`cols` here are the EFFECTIVE
 * numbers (so setting one of them visibly fills in the other), and
 * `layout_notes` carries anything the layout had to do that the rules did not
 * ask for — a panel spilled out of a claimed cell, a grid grown to fit.
 */
interface GridMeta {
  rows?: number
  cols?: number
  panels?: number
  layout_notes?: string[]
}

interface VariantSet {
  name: string | null
  selection: Record<string, unknown>
}

interface Spec {
  measures: string[]
  roles: Record<string, Role>
  kind: string
  aggregate?: { statistic: string; error: string }
  facet?: FacetOptions
  variant_policy?: string
  /* One entry per row of the Variants section. One row is a pin (the figure
     shows that variant); several are a comparison, and a `Variant` factor
     appears in Factors carrying whichever role the user gives it. */
  variant_sets?: VariantSet[]
  style?: Record<string, unknown>
}

interface DescribeResponse {
  catalog: { measures: { name: string; shape: string; plottable: boolean }[] }
  variable: string | null
  eligible?: boolean
  reason?: string | null
  table?: TableInfo
  spec?: Spec
  capabilities?: Capabilities
  joinable_with?: string[]
}

interface FigurePayload {
  key: Record<string, unknown>
  label: string
  figure: { data: unknown[]; layout: Record<string, unknown> }
  row_count: number
  downsampled_from: number | null
}

interface Props {
  /** Variable type (scidb) or column name (CSV). Empty picks the default. */
  variable: string
  /** Set to plot a CSV file instead of the project database. */
  csvPath?: string
  /**
   * True when this IS the whole webview (its own VS Code tab) rather than a
   * modal over the DAG: no backdrop, no rounded card, no close button — the
   * tab's own chrome does that.
   */
  embedded?: boolean
  onClose: () => void
}

export default function PlotStudio({ variable, csvPath, embedded = false, onClose }: Props) {
  // Threaded into every call: the backend picks CsvSource or ScidbSource from
  // it, and nothing else about the panel changes (one DataSource protocol).
  const sourceParams = useMemo(
    () => (csvPath ? { csv_path: csvPath } : {}),
    [csvPath]
  )
  const [describe, setDescribe] = useState<DescribeResponse | null>(null)
  const [spec, setSpec] = useState<Spec | null>(null)
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [figures, setFigures] = useState<FigurePayload[]>([])
  const [specError, setSpecError] = useState('')
  const [loadError, setLoadError] = useState('')
  const [busy, setBusy] = useState(false)
  const [code, setCode] = useState<string | null>(null)
  const [notice, setNotice] = useState('')
  // Reclaiming space happens at two levels: hiding the controls rail inside
  // the panel, and asking VS Code to enlarge the tab the webview lives in.
  const [controlsHidden, setControlsHidden] = useState(false)
  // Index of the variant row whose DAG popup is open, or null.
  const [variantEditor, setVariantEditor] = useState<number | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [canvasHeight, setCanvasHeight] = useState(0)
  const observerRef = useRef<ResizeObserver | null>(null)

  // A CSV has no variable type; name the file instead.
  const title = describe?.variable ?? variable ?? (csvPath ? csvPath.split('/').pop() ?? 'CSV' : '')

  // Plotly needs a definite pixel height, so measure the canvas rather than
  // hardcoding one: a maximized panel should give the figure the extra space.
  //
  // A callback ref, NOT a mount effect: on mount this component renders the
  // "Loading…" branch, which has no canvas, so an effect keyed on [] observed
  // nothing and left the height at 0 — the figure then sat at its 320px floor
  // in a full-height pane.
  const canvasRef = useCallback((node: HTMLDivElement | null) => {
    observerRef.current?.disconnect()
    if (!node || typeof ResizeObserver === 'undefined') return
    setCanvasHeight(node.getBoundingClientRect().height)
    const observer = new ResizeObserver(entries => {
      setCanvasHeight(entries[0].contentRect.height)
    })
    observer.observe(node)
    observerRef.current = observer
  }, [])

  useEffect(() => () => observerRef.current?.disconnect(), [])

  // --- open ---------------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    setDescribe(null)
    setSpec(null)
    setFigures([])
    setLoadError('')
    callBackend('plot_describe', { variable: variable || undefined, ...sourceParams })
      .then(raw => {
        if (cancelled) return
        const response = raw as DescribeResponse
        setDescribe(response)
        if (response.spec) setSpec(response.spec)
        if (response.capabilities) setCapabilities(response.capabilities)
      })
      .catch(err => !cancelled && setLoadError((err as Error).message))
    return () => { cancelled = true }
  }, [variable, sourceParams])

  // --- resolve on every spec change (debounced) ---------------------------
  const timer = useRef<number | null>(null)
  useEffect(() => {
    if (!spec) return
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      setBusy(true)
      Promise.all([
        callBackend('plot_resolve', { spec, ...sourceParams }),
        callBackend('plot_capabilities', { spec, ...sourceParams }).catch(() => null),
      ])
        .then(([resolved, caps]) => {
          const result = resolved as { ok: boolean; error: string | null; figures: FigurePayload[] }
          setSpecError(result.ok ? '' : (result.error ?? 'Could not resolve this plot.'))
          setFigures(result.figures ?? [])
          if (caps) setCapabilities(caps as Capabilities)
        })
        .catch(err => setSpecError((err as Error).message))
        .finally(() => setBusy(false))
    }, 180)
    return () => { if (timer.current) window.clearTimeout(timer.current) }
  }, [spec, sourceParams])

  // --- spec edits ---------------------------------------------------------
  const setRole = useCallback((factor: string, role: Role) => {
    setSpec(prev => (prev ? { ...prev, roles: { ...prev.roles, [factor]: role } } : prev))
  }, [])

  const setKind = useCallback((kind: string) => {
    setSpec(prev => (prev ? { ...prev, kind } : prev))
  }, [])

  const setAggregate = useCallback((patch: { statistic?: string; error?: string }) => {
    setSpec(prev => prev && ({
      ...prev,
      aggregate: { statistic: 'mean', error: 'sd', ...(prev.aggregate ?? {}), ...patch },
    }))
  }, [])

  const setFacet = useCallback((patch: Partial<FacetOptions>) => {
    setSpec(prev => (prev ? { ...prev, facet: { ...(prev.facet ?? {}), ...patch } } : prev))
  }, [])

  /**
   * Edit the rule in one grid slot, padding the array out to reach it.
   *
   * The slots ARE the grid — there is no add/remove, because the number of rows
   * and columns is the thing the user set. A slot left blank claims nothing and
   * takes whatever is left over, in order (scistackplot's Matcher.is_blank).
   */
  const setRuleAt = useCallback(
    (axis: 'rows' | 'cols', index: number, patch: Partial<Matcher>) => {
      setSpec(prev => {
        if (!prev) return prev
        const rules = [...(prev.facet?.[axis] ?? [])]
        while (rules.length <= index) rules.push({ op: 'contains', value: '' })
        rules[index] = { ...rules[index], ...patch }
        return { ...prev, facet: { ...(prev.facet ?? {}), [axis]: rules } }
      })
    },
    []
  )

  /** Average across variant factors no row selected. The deliberate opt-in that
   *  `roles.validate` demands — pooling looks exactly like not pooling. */
  const setPooling = useCallback((pool: boolean) => {
    setSpec(prev => (prev ? { ...prev, variant_policy: pool ? 'pool' : 'facet' } : prev))
  }, [])

  /** Add a row. Its selection starts empty — "all variants" — because the honest
   *  starting point for a new comparison is everything, narrowed on the DAG. */
  const addVariantSet = useCallback(() => {
    setSpec(prev =>
      prev
        ? { ...prev, variant_sets: [...(prev.variant_sets ?? []), { name: null, selection: {} }] }
        : prev
    )
  }, [])

  const removeVariantSet = useCallback((index: number) => {
    setSpec(prev =>
      prev
        ? { ...prev, variant_sets: (prev.variant_sets ?? []).filter((_, i) => i !== index) }
        : prev
    )
  }, [])

  const editVariantSet = useCallback((index: number, patch: Partial<VariantSet>) => {
    setSpec(prev => {
      if (!prev) return prev
      const sets = [...(prev.variant_sets ?? [])]
      if (!sets[index]) return prev
      sets[index] = { ...sets[index], ...patch }
      return { ...prev, variant_sets: sets }
    })
  }, [])

  // Factors of the resolved table (Variant included, selected-away columns
  // gone). `describe.table.factors` is the pre-selection view and would show
  // both a `Code:` column and the `Variant` factor that consumed it.
  const factors = capabilities?.factors ?? describe?.table?.factors ?? []

  // Rows come from the SPEC, annotations from the backend. The spec is the
  // source of truth and updates on the keystroke; `capabilities` is a debounced
  // echo, so rendering rows from it would make "+ Add variant" appear to do
  // nothing for a moment — and, worse, would let a row index mean different
  // things in the list and in the popup during that window.
  const variantRows = useMemo(
    () =>
      (spec?.variant_sets ?? []).map((set, index) => {
        const info = capabilities?.variants?.sets?.[index]
        return {
          explicitName: set.name,
          autoLabel: info?.auto_label ?? '(not set)',
          defined: Object.keys(set.selection ?? {}).length > 0,
          rowCount: info?.row_count,
          spans: info?.spans ?? {},
        }
      }),
    [spec?.variant_sets, capabilities]
  )
  // The section exists whenever this source has variants at all — a project
  // with none never sees it, and one that does always has a row to edit.
  const hasVariants = Boolean(
    (capabilities?.variants?.factors ?? []).length > 0 || variantRows.length > 0
  )
  const summarizing = spec?.kind === 'bar' || spec?.kind === 'band'
  const faceted = Object.values(spec?.roles ?? {}).includes('facet')

  // The grid the backend actually built. Read back rather than recomputed: the
  // "set one, the other follows" arithmetic is grid_shape_for's, in
  // scistackplot, and a second copy of it here would be the thing that drifts.
  const gridMeta = (figures[0]?.figure?.layout?.meta ?? {}) as GridMeta
  const effRows = Math.max(1, gridMeta.rows ?? 1)
  const effCols = Math.max(1, gridMeta.cols ?? 1)
  const layoutNotes = gridMeta.layout_notes ?? []

  // --- export -------------------------------------------------------------
  const handleExport = useCallback(() => {
    if (!spec) return
    setNotice('')
    callBackend('plot_export', { spec, ...sourceParams })
      .then(raw => setCode((raw as { source: string }).source))
      .catch(err => setNotice(`Export failed: ${(err as Error).message}`))
  }, [spec, sourceParams])

  const handleSave = useCallback(async () => {
    if (!spec) return
    setNotice('')
    const defaultName = `${title || 'figure'}.png`.replace(/[^\w.-]+/g, '_')
    try {
      let path: string | null = null
      if (isVSCodeMode) {
        const picked = await callBackend('pick_save_path', { defaultName })
        path = (picked as { path: string | null }).path
        if (!path) return  // dialog cancelled
      } else {
        path = window.prompt('Save the figure as:', defaultName)
        if (!path) return
      }
      setNotice('Rendering at full resolution…')
      const result = (await callBackend('plot_save_figure', { spec, path, ...sourceParams })) as {
        ok: boolean
        error: string | null
        files: string[]
      }
      if (!result.ok) setNotice(`Could not save: ${result.error}`)
      else setNotice(`Saved ${result.files.join(', ')}`)
    } catch (err) {
      setNotice(`Could not save: ${(err as Error).message}`)
    }
  }, [spec, sourceParams, title])

  const handleAddToPipeline = useCallback(() => {
    if (!spec) return
    setNotice('Writing endpoint…')
    callBackend('plot_add_to_pipeline', { spec })
      .then(raw => {
        const result = raw as { ok?: boolean; error?: string; function_name?: string; file?: string }
        if (result.error) setNotice(`Could not add: ${result.error}`)
        else setNotice(`Added ${result.function_name} to ${result.file}. Wire it up on the canvas.`)
      })
      .catch(err => setNotice(`Could not add: ${(err as Error).message}`))
  }, [spec])

  // --- render -------------------------------------------------------------
  if (loadError) {
    return (
      <Shell variable={title} shape={capabilities?.shape} onClose={onClose} embedded={embedded}>
        <div style={styles.error}>Could not open the plot panel: {loadError}</div>
      </Shell>
    )
  }
  if (!describe) {
    return (
      <Shell variable={title} shape={capabilities?.shape} onClose={onClose} embedded={embedded}>
        <div style={styles.note}>Loading…</div>
      </Shell>
    )
  }
  if (describe.eligible === false) {
    // The empty state the design doc insists on: say why, never draw blank axes.
    return (
      <Shell variable={title} shape={capabilities?.shape} onClose={onClose} embedded={embedded}>
        <div style={styles.note}>{describe.reason}</div>
      </Shell>
    )
  }

  const gridRows = (figures[0]?.figure?.layout?.meta as { rows?: number } | undefined)?.rows ?? 1
  const available = figures.length > 1
    ? Math.round(canvasHeight / 2) - 28
    : canvasHeight - 16
  // Fill the pane, but never squeeze a tall facet grid: each row needs room for
  // its own tick labels and the next row's title.
  const figureHeight = Math.max(320, gridRows * 240, available)

  return (
    <Shell
      variable={title}
      // The measure's shape used to head its own sidebar section, which spent a
      // whole block restating the title. It belongs to the title.
      shape={capabilities?.shape}
      onClose={onClose}
      embedded={embedded}
      panelRef={panelRef}
      controlsHidden={controlsHidden}
      onToggleControls={() => setControlsHidden(v => !v)}
      sidebar={
        <div
          style={{
            ...styles.controls,
            ...(controlsHidden ? styles.controlsHidden : null),
          }}
        >
          {hasVariants && (
            <Section title="Variants">
              <div style={styles.hint}>
                Each row is one variant of the pipeline. Two or more become a
                factor you can colour or facet by.
              </div>
              <VariantRows
                rows={variantRows}
                onRename={(index, name) => editVariantSet(index, { name })}
                onEdit={index => setVariantEditor(index)}
                onRemove={removeVariantSet}
                onAdd={addVariantSet}
              />
              <label style={styles.poolRow} title="Average across variant factors no row selected — results from different pipeline variants are combined">
                <input
                  type="checkbox"
                  checked={spec?.variant_policy === 'pool'}
                  onChange={e => setPooling(e.target.checked)}
                  style={{ marginRight: 6 }}
                />
                Pool unselected variants
              </label>
              <VariantReadout summary={capabilities?.variants} />
            </Section>
          )}

          <Section title="Factors">
            <div style={styles.hint}>Each factor does exactly one thing.</div>
            {factors.map(factor => (
              <label key={factor.name} style={styles.factorRow}>
                <span style={styles.factorName}>
                  {factor.display}
                  {factor.is_variant && <span style={styles.variantTag} title="A pipeline variant, not a replicate">variant</span>}
                  {factor.is_field && <span style={styles.fieldTag} title="The fields of this struct/dict variable — one subplot each by default">fields</span>}
                  <span style={styles.levelCount}>{factor.level_count}</span>
                </span>
                <select
                  value={spec?.roles?.[factor.name] ?? 'free'}
                  onChange={e => setRole(factor.name, e.target.value as Role)}
                  style={styles.select}
                >
                  {ROLE_OPTIONS.map(option => (
                    <option key={option.value} value={option.value} title={option.hint}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </Section>

          <Section title="Plot type">
            {(capabilities?.kinds ?? []).map(info => (
              <label
                key={info.kind}
                style={{ ...styles.kindRow, opacity: info.available ? 1 : 0.45 }}
                title={info.reason ?? ''}
              >
                <input
                  type="radio"
                  name="plot-kind"
                  checked={spec?.kind === info.kind}
                  disabled={!info.available}
                  onChange={() => setKind(info.kind)}
                  style={{ marginRight: 6 }}
                />
                {KIND_LABELS[info.kind] ?? info.kind}
              </label>
            ))}
          </Section>

          {faceted && (
            <Section title="Layout">
              <div style={styles.hint}>
                Set rows or columns — the other follows from the number of
                subplots. Then name what belongs in each one; blank takes
                whatever is left, in order.
              </div>
              <div style={styles.gridSizeRow}>
                <GridSizeInput
                  label="N rows"
                  value={spec?.facet?.n_rows ?? null}
                  effective={effRows}
                  onChange={n => setFacet({ n_rows: n })}
                />
                <GridSizeInput
                  label="N columns"
                  value={spec?.facet?.n_cols ?? null}
                  effective={effCols}
                  onChange={n => setFacet({ n_cols: n })}
                />
              </div>

              <RuleSlots
                title="Rows"
                count={effRows}
                rules={spec?.facet?.rows ?? []}
                onEdit={(i, patch) => setRuleAt('rows', i, patch)}
              />
              <RuleSlots
                title="Columns"
                count={effCols}
                rules={spec?.facet?.cols ?? []}
                onEdit={(i, patch) => setRuleAt('cols', i, patch)}
              />

              {/* Never let the layout quietly disobey a rule: if a subplot had
                  to move, or the grid had to grow, say which and why. */}
              {layoutNotes.map((note, index) => (
                <div key={index} style={styles.layoutNote}>{note}</div>
              ))}
            </Section>
          )}

          {summarizing && (
            <Section title="Summary">
              <label style={styles.factorRow}>
                <span style={styles.factorName}>Centre</span>
                <select
                  value={spec?.aggregate?.statistic ?? 'mean'}
                  onChange={e => setAggregate({ statistic: e.target.value })}
                  style={styles.select}
                >
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                </select>
              </label>
              <label style={styles.factorRow}>
                <span style={styles.factorName}>Spread</span>
                <select
                  value={spec?.aggregate?.error ?? 'sd'}
                  onChange={e => setAggregate({ error: e.target.value })}
                  style={styles.select}
                >
                  <option value="sd">SD</option>
                  <option value="sem">SEM</option>
                  <option value="ci95">95% CI</option>
                  <option value="iqr">IQR</option>
                  <option value="none">None</option>
                </select>
              </label>
            </Section>
          )}

          <div style={{ ...styles.actions, flexWrap: 'wrap' }}>
            <button
              type="button"
              style={styles.button}
              onClick={handleSave}
              title="Render with matplotlib at full resolution — the same figure the pipeline would produce"
            >
              Save image
            </button>
            <button type="button" style={styles.button} onClick={handleExport}>
              Export code
            </button>
            <button type="button" style={styles.primaryButton} onClick={handleAddToPipeline}>
              Add to pipeline
            </button>
          </div>
          {notice && <div style={styles.notice}>{notice}</div>}
        </div>
      }
    >
      <div style={styles.canvas} ref={canvasRef}>
        {specError && <div style={styles.specError}>{specError}</div>}
        {busy && <div style={styles.note}>Resolving…</div>}
        {!specError && figures.length === 0 && !busy && (
          <div style={styles.note}>Nothing to plot with these settings.</div>
        )}
        {figures.map((figure, index) => (
          <div
            key={figure.label || index}
            style={{
              ...styles.figureBlock,
              ...(figures.length === 1 ? { height: '100%' } : null),
            }}
          >
            {figure.label && <div style={styles.figureLabel}>{figure.label}</div>}
            {figure.downsampled_from && (
              <div style={styles.downsampleNote}>
                Showing a reduced view of {figure.downsampled_from.toLocaleString()} points —
                the exported figure uses every point.
              </div>
            )}
            <Plot
              data={figure.figure.data}
              layout={{
                ...figure.figure.layout,
                autosize: true,
                height: figureHeight,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#ccc', size: 11 },
              }}
              config={{
                displaylogo: false,
                responsive: true,
                // A webview cannot download; "Save image" does it server-side.
                modeBarButtonsToRemove: isVSCodeMode ? ['toImage'] : [],
              }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          </div>
        ))}
      </div>

      {code !== null && (
        <div style={styles.codeOverlay} onClick={() => setCode(null)}>
          <pre style={styles.code} onClick={e => e.stopPropagation()}>{code}</pre>
        </div>
      )}

      {variantEditor !== null && !csvPath && (
        <VariantDagPopup
          variable={describe?.variable ?? variable}
          selection={(spec?.variant_sets?.[variantEditor]?.selection ?? {}) as Record<string, unknown>}
          name={variantRows[variantEditor]?.explicitName ?? ''}
          placeholder={variantRows[variantEditor]?.autoLabel ?? ''}
          onCancel={() => setVariantEditor(null)}
          onApply={({ selection, name }) => {
            // An empty name stays null so the label keeps following the
            // selection; typing one pins it.
            editVariantSet(variantEditor, { selection, name: name || null })
            setVariantEditor(null)
          }}
        />
      )}
    </Shell>
  )
}

interface ShellProps {
  variable: string
  /** The measure's shape (`scalar`, `series_1d`, …), badged beside the title. */
  shape?: string
  onClose: () => void
  embedded?: boolean
  /** The controls rail. It sits under the header, in the same narrow column. */
  sidebar?: React.ReactNode
  /** The figure area — it owns the full height of the panel. */
  children: React.ReactNode
  panelRef?: React.RefObject<HTMLDivElement>
  controlsHidden?: boolean
  onToggleControls?: () => void
}

function Shell({
  variable,
  shape,
  onClose,
  embedded,
  sidebar,
  children,
  panelRef,
  controlsHidden,
  onToggleControls,
}: ShellProps) {
  return (
    // No overlay click-to-close: a stray click on the backdrop while dragging a
    // plotly selection would throw the panel away mid-exploration.
    <div style={embedded ? styles.embeddedRoot : styles.overlay}>
      {/* A row, not a column: the title bar caps the controls rail rather than
          spanning the panel, so the figure keeps the tab's full height. */}
      <div ref={panelRef} style={embedded ? styles.embeddedPanel : styles.panel}>
        <div style={{ ...styles.rail, ...(controlsHidden ? styles.railCollapsed : null) }}>
          <div style={styles.header}>
            {/* The toggle sits over the rail it collapses, so the control is
                where the thing it controls is. */}
            <div style={styles.headerLeft}>
              {onToggleControls && (
                <button
                  type="button"
                  style={styles.headerButton}
                  onClick={onToggleControls}
                  title={controlsHidden ? 'Show the controls' : "Give the figure the controls' width"}
                >
                  {controlsHidden ? '❯' : '❮'}
                </button>
              )}
              {/* A collapsed rail is only as wide as its buttons, so the title
                  would be the one thing keeping it wide. */}
              {!controlsHidden && (
                <span style={styles.title} title={`Plot — ${variable}`}>
                  Plot — {variable}
                  {shape && <span style={styles.shapeTag}>{shape}</span>}
                </span>
              )}
            </div>
            <div style={styles.headerActions}>
              {!embedded && (
                <button type="button" style={styles.close} onClick={onClose}>✕</button>
              )}
            </div>
          </div>
          {sidebar}
        </div>
        {children}
      </div>
    </div>
  )
}

interface GridSizeInputProps {
  label: string
  /** What the user pinned, or null when this dimension is computed. */
  value: number | null
  /** What the backend laid out — shown as a placeholder while unpinned. */
  effective: number
  onChange: (value: number | null) => void
}

/**
 * One dimension of the facet grid.
 *
 * An empty box is not "1", it is "you decide": the backend computes it from the
 * other dimension and the subplot count, and the result shows through as the
 * placeholder. That is what makes "I set 2 columns" answer "so, 3 rows".
 */
function GridSizeInput({ label, value, effective, onChange }: GridSizeInputProps) {
  return (
    <label style={styles.gridSizeField}>
      <span style={styles.gridSizeLabel}>{label}</span>
      <input
        type="number"
        min={1}
        value={value ?? ''}
        placeholder={String(effective)}
        onChange={e => onChange(e.target.value ? Number(e.target.value) : null)}
        title={value === null ? `Computed: ${effective}. Type a number to pin it.` : 'Clear to compute automatically'}
        style={{
          ...styles.select,
          width: 56,
          ...(value === null ? styles.gridSizeAuto : null),
        }}
      />
    </label>
  )
}

interface RuleSlotsProps {
  title: string
  /** Number of slots = the grid dimension. Comes from the rendered layout. */
  count: number
  rules: Matcher[]
  onEdit: (index: number, patch: Partial<Matcher>) => void
}

/**
 * One axis of the facet grid, one editor per slot.
 *
 * Fixed length, not a list with +/- buttons: the grid already has a known
 * number of rows and columns, so a rule is a property OF row 2, not an extra
 * row. Each slot claims the panels whose name it matches; a blank slot takes
 * whatever is left over, in order, and anything matching no slot at all lands
 * in a trailing "other" row/column rather than disappearing.
 */
function RuleSlots({ title, count, rules, onEdit }: RuleSlotsProps) {
  const singular = title.replace(/s$/, '')
  return (
    <div style={styles.ruleList}>
      <div style={styles.ruleTitle}>{title}</div>
      {Array.from({ length: count }, (_, index) => {
        const rule = rules[index] ?? { op: 'contains' as MatchOp, value: '' }
        return (
          <div key={index} style={styles.ruleRow}>
            <span style={styles.slotIndex}>{singular} {index + 1}</span>
            <select
              value={rule.op}
              onChange={e => onEdit(index, { op: e.target.value as MatchOp })}
              style={{ ...styles.select, flex: '0 0 88px' }}
            >
              {MATCH_OPS.map(op => (
                <option key={op.value} value={op.value}>{op.label}</option>
              ))}
            </select>
            <input
              value={rule.value}
              onChange={e => onEdit(index, { value: e.target.value })}
              placeholder="anything"
              style={{ ...styles.select, flex: 1, minWidth: 0 }}
            />
          </div>
        )
      })}
    </div>
  )
}

interface VariantRow {
  /** The user's name, or null while the label follows the selection. */
  explicitName: string | null
  /** What the selection says it is, used as the placeholder. */
  autoLabel: string
  /** False until the row selects something. Such a row changes nothing. */
  defined: boolean
  /** Rows it contributes, once the backend has measured it. */
  rowCount?: number
  /** Code axes it leaves open and disagrees on, `{column: version count}`. */
  spans: Record<string, number>
}

interface VariantRowsProps {
  rows: VariantRow[]
  onRename: (index: number, name: string | null) => void
  onEdit: (index: number) => void
  onRemove: (index: number) => void
  onAdd: () => void
}

/**
 * The Variants section: a name and a "select on the DAG" button, per row.
 *
 * Shaped like Factors — two columns, many rows — because it is the same kind of
 * list: a thing, and what to do with it. The name is editable and lands in the
 * figure's legend, which is the whole reason a variant is *named* rather than
 * numbered: "baseline" and "20 Hz filter" survive being read a week later,
 * "variant 2" does not.
 *
 * The selection itself is not editable here. It is a point in a space whose
 * coordinates are pipeline nodes, and the pipeline canvas is already the
 * picture of that space — see VariantDagPopup.
 */
function VariantRows({ rows, onRename, onEdit, onRemove, onAdd }: VariantRowsProps) {
  return (
    <div style={styles.variantPicker}>
      {rows.map((row, index) => (
        <div key={index} style={styles.variantSetRow}>
          <input
            value={row.explicitName ?? ''}
            placeholder={row.autoLabel}
            onChange={e => onRename(index, e.target.value || null)}
            style={styles.variantNameInput}
            title={
              row.explicitName === null
                ? `Named from its selection: ${row.autoLabel}. Type to override.`
                : 'The name this variant carries in the figure'
            }
          />
          <button
            type="button"
            style={styles.variantSelectButton}
            onClick={() => onEdit(index)}
            title="Choose this variant on the pipeline graph"
          >
            ⋔ Select
          </button>
          {/* One variant is the pin; there is nothing to remove down to zero. */}
          {rows.length > 1 && (
            <button
              type="button"
              style={styles.variantRemove}
              onClick={() => onRemove(index)}
              title="Remove this variant"
            >
              ✕
            </button>
          )}
          {/* An unfilled row is inert, not broken: it must not wear the same
              warning as a selection that genuinely matched nothing. */}
          {!row.defined && (
            <span style={styles.variantUnsetTag} title="Nothing selected yet — this variant does not affect the figure. Click Select.">
              not set
            </span>
          )}
          {row.defined && row.rowCount === 0 && (
            <span style={styles.variantEmptyTag} title="This selection matched no records — check it against what has actually run">
              no data
            </span>
          )}
          {Object.keys(row.spans).length > 0 && (
            <span
              style={styles.variantEmptyTag}
              title={
                `This variant pools ${Object.entries(row.spans)
                  .map(([column, n]) => `${n} versions of ${column}`)
                  .join(', ')} — its rows were built by more than one version ` +
                `of that code. Pin a version on it, or split it into one variant per version.`
              }
            >
              pools {Object.values(row.spans)[0]} versions
            </span>
          )}
        </div>
      ))}
      <button
        type="button"
        style={styles.variantAdd}
        onClick={onAdd}
        title="Compare against another variant"
      >
        + Add variant
      </button>
    </div>
  )
}

/**
 * How much of the variant space is on screen.
 *
 * The readout matters as much as the rows. A canvas or a factor list shows
 * coordinates; neither shows the PRODUCT, and an exploding panel count is the
 * thing that actually catches people out.
 */
function VariantReadout({ summary }: { summary?: VariantSummary }) {
  if (!summary || summary.factors.length === 0) return null
  const { total_combinations, selected_combinations } = summary
  const none = selected_combinations === 0
  return (
    <div style={none ? styles.variantCountEmpty : styles.variantCount}>
      {none
        ? 'Nothing selected — the figure would be empty.'
        : `${selected_combinations} of ${total_combinations} variant combination${
            total_combinations === 1 ? '' : 's'
          }`}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={styles.section}>
      <div style={styles.sectionTitle}>{title}</div>
      {children}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  // Own-tab mode: fill the webview exactly, no card, no backdrop.
  embeddedRoot: { position: 'absolute', inset: 0, background: '#16162a' },
  embeddedPanel: {
    width: '100%', height: '100%', background: '#16162a',
    display: 'flex', flexDirection: 'row', overflow: 'hidden',
  },
  panel: {
    width: '96vw', height: '94vh', background: '#16162a',
    border: '1px solid #3a3a5a', borderRadius: 8,
    display: 'flex', flexDirection: 'row', overflow: 'hidden',
  },
  // The title bar and the controls share one column, so the figure column
  // starts at the very top of the panel.
  rail: {
    width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column',
    minHeight: 0, borderRight: '1px solid #2a2a4a',
  },
  // Collapsed: only as wide as the toggle that brings it back.
  railCollapsed: { width: 'auto' },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 6, padding: '8px 12px', borderBottom: '1px solid #2a2a4a',
    background: '#1a1a2e', flexShrink: 0,
  },
  title: {
    color: '#eee', fontSize: 13, fontWeight: 600,
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  close: {
    background: 'transparent', border: 'none', color: '#888',
    cursor: 'pointer', fontSize: 14,
  },
  // LONGHANDS only, and the same keys in both states — see controlsHidden.
  controls: {
    padding: 12, flex: 1, minHeight: 0, overflowX: 'hidden', overflowY: 'auto',
  },
  // Collapsed rather than unmounted: the control state survives the toggle.
  //
  // These two objects must name exactly the same overflow properties. React
  // removes whatever a re-render drops, so a collapsed state that added the
  // `overflow` SHORTHAND left React clearing `overflow` on reopen — and
  // clearing the shorthand clears overflow-y with it, while the unchanged
  // `overflowY: 'auto'` was skipped as "nothing to re-apply". The rail came
  // back unscrollable, and only after a collapse/expand cycle.
  controlsHidden: {
    width: 0, flex: '0 0 0', padding: 0, overflowX: 'hidden', overflowY: 'hidden',
  },
  headerActions: { display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 },
  ruleList: { marginTop: 6 },
  ruleTitle: { fontSize: 10, color: '#999', marginBottom: 3 },
  ruleRow: { display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 },
  slotIndex: { fontSize: 10, color: '#888', flex: '0 0 52px' },
  gridSizeRow: { display: 'flex', gap: 8, marginBottom: 4 },
  gridSizeField: { display: 'flex', alignItems: 'center', gap: 4 },
  gridSizeLabel: { fontSize: 11, color: '#bbb' },
  // An unpinned dimension reads as derived, not as something the user typed.
  gridSizeAuto: { color: '#8a8aa8', fontStyle: 'italic' },
  layoutNote: {
    fontSize: 10, color: '#e0b050', marginTop: 6, lineHeight: 1.4,
  },
  headerButton: {
    background: '#22223a', color: '#ccc', border: '1px solid #3a3a5a',
    borderRadius: 4, cursor: 'pointer', fontSize: 11, padding: '2px 8px',
  },
  // minWidth 0: a flex item defaults to its content's width, and a wide plotly
  // figure would then push the rail off the panel instead of scrolling.
  canvas: { flex: 1, minWidth: 0, padding: 12, overflowY: 'auto' },
  section: { marginBottom: 16 },
  sectionTitle: {
    fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6,
    color: '#7b68ee', marginBottom: 6, fontWeight: 700,
  },
  hint: { fontSize: 10, color: '#777', marginBottom: 6, fontStyle: 'italic' },
  pinNote: {
    fontSize: 10, lineHeight: 1.45, color: '#d9c48f', marginBottom: 6,
    padding: '5px 7px', background: '#221c0c', borderLeft: '2px solid #d9b45f',
  },
  factorRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 6, marginBottom: 5,
  },
  factorName: {
    fontSize: 11, fontFamily: 'monospace', color: '#ccc',
    display: 'flex', alignItems: 'center', gap: 4, minWidth: 0,
  },
  levelCount: {
    fontSize: 9, color: '#666', background: '#22223a',
    borderRadius: 8, padding: '0 5px',
  },
  variantTag: {
    fontSize: 8, color: '#fbbf24', border: '1px solid #6b5a1a',
    borderRadius: 3, padding: '0 3px', textTransform: 'uppercase',
  },
  fieldTag: {
    fontSize: 8, color: '#67e8f9', border: '1px solid #1a5a6b',
    borderRadius: 3, padding: '0 3px', textTransform: 'uppercase',
  },
  codeTag: {
    fontSize: 8, color: '#c4b5fd', border: '1px solid #4c3a8a',
    borderRadius: 3, padding: '0 3px', textTransform: 'uppercase',
    marginLeft: 6,
  },
  variantPicker: {
    display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8,
  },
  variantSetRow: {
    display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap',
  },
  variantNameInput: {
    flex: 1, minWidth: 0,
    background: '#22223a', color: '#ddd', border: '1px solid #3a3a5a',
    borderRadius: 4, fontSize: 11, padding: '3px 5px',
  },
  variantSelectButton: {
    flex: '0 0 auto', padding: '3px 8px', background: '#22223a', color: '#c4b5fd',
    border: '1px solid #4c3a8a', borderRadius: 4, cursor: 'pointer', fontSize: 11,
  },
  variantRemove: {
    flex: '0 0 auto', padding: '2px 5px', background: 'transparent', color: '#888',
    border: 'none', cursor: 'pointer', fontSize: 11,
  },
  variantEmptyTag: {
    fontSize: 9, color: '#fbbf24', border: '1px solid #6b5a1a',
    borderRadius: 3, padding: '0 3px', textTransform: 'uppercase',
  },
  // Grey, not amber: "not yet said" is a state, not a problem.
  variantUnsetTag: {
    fontSize: 9, color: '#8a8aa8', border: '1px solid #3a3a5a',
    borderRadius: 3, padding: '0 3px', textTransform: 'uppercase',
  },
  variantAdd: {
    alignSelf: 'flex-start', padding: '3px 10px', background: 'transparent',
    color: '#9d92f5', border: '1px dashed #4c3a8a', borderRadius: 4,
    cursor: 'pointer', fontSize: 11,
  },
  poolRow: {
    display: 'flex', alignItems: 'center', fontSize: 11, color: '#bbb',
    marginTop: 8, cursor: 'pointer',
  },
  variantRow: { display: 'flex', flexDirection: 'column', gap: 2 },
  variantName: {
    fontSize: 10, color: '#9ca3af', fontFamily: 'monospace',
    display: 'flex', alignItems: 'center',
  },
  variantLevels: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  variantLevel: {
    display: 'flex', alignItems: 'center', gap: 3,
    fontSize: 11, color: '#ddd', cursor: 'pointer',
  },
  variantCount: {
    fontSize: 10, color: '#9ca3af', marginTop: 4,
    borderTop: '1px solid #333', paddingTop: 4,
  },
  variantCountEmpty: {
    fontSize: 10, color: '#fbbf24', marginTop: 4,
    borderTop: '1px solid #333', paddingTop: 4,
  },
  shapeTag: { fontSize: 9, color: '#67e8f9', marginLeft: 6 },
  readonlyValue: { fontSize: 12, fontFamily: 'monospace', color: '#eee' },
  select: {
    background: '#22223a', color: '#ddd', border: '1px solid #3a3a5a',
    borderRadius: 4, fontSize: 11, padding: '2px 4px', maxWidth: 130,
  },
  kindRow: {
    display: 'flex', alignItems: 'center', fontSize: 11,
    color: '#ccc', marginBottom: 3, cursor: 'pointer',
  },
  actions: { display: 'flex', gap: 6, marginTop: 8 },
  button: {
    flex: 1, padding: '5px 8px', background: '#22223a', color: '#ccc',
    border: '1px solid #3a3a5a', borderRadius: 4, cursor: 'pointer', fontSize: 11,
  },
  primaryButton: {
    flex: 1, padding: '5px 8px', background: '#7b68ee', color: '#fff',
    border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 11, fontWeight: 600,
  },
  notice: { fontSize: 10, color: '#67e8f9', marginTop: 6 },
  note: { fontSize: 12, color: '#777', fontStyle: 'italic', padding: 8 },
  error: { fontSize: 12, color: '#f87171', padding: 12 },
  specError: {
    fontSize: 11, color: '#fbbf24', background: '#2a2416',
    border: '1px solid #6b5a1a', borderRadius: 4, padding: 8, marginBottom: 8,
  },
  figureBlock: { marginBottom: 14 },
  figureLabel: {
    fontSize: 11, fontFamily: 'monospace', color: '#aaa', marginBottom: 2,
  },
  downsampleNote: { fontSize: 10, color: '#fbbf24', marginBottom: 4 },
  codeOverlay: {
    position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.75)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40,
  },
  code: {
    background: '#0e0e1a', color: '#ddd', border: '1px solid #3a3a5a',
    borderRadius: 6, padding: 16, fontSize: 11, fontFamily: 'monospace',
    maxHeight: '80%', maxWidth: '80%', overflow: 'auto', whiteSpace: 'pre',
  },
}
