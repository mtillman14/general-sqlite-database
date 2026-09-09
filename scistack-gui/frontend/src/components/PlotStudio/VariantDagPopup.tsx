/**
 * VariantDagPopup — pick a variant on the pipeline graph.
 *
 * The same DAG the SciStack Pipeline canvas draws, in a modal, wired to a
 * PlotSpec instead of to execution state. Parameter nodes offer their recorded
 * levels as checkboxes; function nodes offer their recorded versions as a
 * dropdown defaulting to "latest".
 *
 * **Why the whole pipeline and not just the relevant part.** The induced
 * subgraph (ancestors of the measure that contribute >1 level) is smaller and
 * was the original design, but it is a graph the user has never seen. Drawing
 * the canvas they already know — with the nodes that cannot matter dimmed —
 * keeps "where am I" free, and makes the absence of a control informative:
 * a greyed node is telling you it does not distinguish these records.
 *
 * **Why a modal and not a tab.** Selection is a decision you come back from,
 * and the state being edited belongs to the panel underneath. Covering it is
 * the point: the popup is doing something different from the canvas of the same
 * shape sitting in another tab, and looking different is how a user knows.
 *
 * Node components are reused verbatim (VariantSelectionContext switches their
 * mode) so this stays a mirror of the canvas as the canvas changes, rather than
 * a copy that slowly drifts from it.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  Handle,
  Position,
  type Edge,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import VariableNode from '../DAG/VariableNode'
import FunctionNode from '../DAG/FunctionNode'
import ParameterNode from '../DAG/ParameterNode'
import PathInputNode from '../DAG/PathInputNode'
import GlueNode from '../DAG/GlueNode'
import { applyDagreLayout } from '../../layout'
import { callBackend } from '../../api'
import {
  VariantSelectionProvider,
  type FunctionVersion,
  type VariantAxis,
  type VariantSelectionValue,
} from '../../context/VariantSelectionContext'

/**
 * A nested pipeline, drawn but not enterable.
 *
 * The real `PipelineNode` reaches for `usePlanRun` and `useScope` so it can run
 * and descend — neither of which exists here (the Plot Studio tab mounts no
 * providers), and neither of which means anything in a selection dialog. Its
 * insides are a different scope; anything selectable in there reaches the user
 * through the "not on this canvas" list below the graph instead.
 */
function InertPipelineNode({ data }: { data: { label?: string } }) {
  return (
    <div style={styles.inertNode} title="A nested pipeline — open it on the canvas to see inside">
      <Handle type="target" position={Position.Left} />
      <div style={styles.inertNodeLabel}>{data.label ?? 'pipeline'}</div>
      <div style={styles.inertNodeHint}>nested pipeline</div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = {
  variableNode: VariableNode,
  functionNode: FunctionNode,
  glueNode: GlueNode,
  parameterNode: ParameterNode,
  pathInputNode: PathInputNode,
  pipelineNode: InertPipelineNode,
}

interface VariantGraph {
  axes: VariantAxis[]
  versions: Record<string, FunctionVersion[]>
  chain_functions: string[]
  /** Name of the per-row "newest at my own schema location" flag. The opening
   *  variant selects on it, and any explicit choice here has to clear it. */
  latest_column: string
}

interface Props {
  /** The measure being plotted — the axes are its. */
  variable: string
  /** Selection being edited, `{column: level | level[] | 'latest'}`. */
  selection: Record<string, unknown>
  /** Row label, editable here too so the popup is self-contained. Empty means
   *  "still following the selection" — `placeholder` is what that resolves to. */
  name: string
  placeholder?: string
  onApply: (next: { selection: Record<string, unknown>; name: string }) => void
  onCancel: () => void
}

export default function VariantDagPopup({
  variable,
  selection: initial,
  name: initialName,
  placeholder,
  onApply,
  onCancel,
}: Props) {
  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [graph, setGraph] = useState<VariantGraph | null>(null)
  const [selection, setSelection] = useState<Record<string, unknown>>(initial)
  const [name, setName] = useState(initialName)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  // --- the graph ----------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const pipeline = (await callBackend('get_pipeline', { pipeline_id: 'main' })) as {
          nodes: Node[]
          edges: Edge[]
        }
        const layout = (await callBackend('get_layout', { pipeline_id: 'main' })) as
          Record<string, unknown>
        const saved = (layout.positions ?? layout) as Record<string, { x: number; y: number }>
        const functions = pipeline.nodes
          .filter(n => n.type === 'functionNode')
          .map(n => (n.data as { label: string }).label)
        const variantGraph = (await callBackend('plot_variant_graph', {
          variable,
          functions,
        })) as VariantGraph
        if (cancelled) return

        // Which functions each parameter feeds — branch params are namespaced
        // per producing function, so "low_hz" alone can match two axes.
        const consumers = new Map<string, string[]>()
        const labelOf = new Map(pipeline.nodes.map(n => [n.id, (n.data as { label?: string }).label ?? '']))
        const typeOf = new Map(pipeline.nodes.map(n => [n.id, n.type]))
        for (const edge of pipeline.edges) {
          if (typeOf.get(edge.source) !== 'parameterNode') continue
          if (typeOf.get(edge.target) !== 'functionNode') continue
          const key = labelOf.get(edge.source) ?? ''
          consumers.set(key, [...(consumers.get(key) ?? []), labelOf.get(edge.target) ?? ''])
        }

        const prepared = pipeline.nodes.map(node =>
          node.type === 'parameterNode'
            ? {
                ...node,
                data: {
                  ...node.data,
                  variantConsumers: consumers.get((node.data as { label: string }).label) ?? [],
                },
              }
            : node
        )
        setNodes(applyDagreLayout(prepared, pipeline.edges, saved))
        setEdges(pipeline.edges)
        setGraph(variantGraph)
      } catch (err) {
        if (!cancelled) setError((err as Error).message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [variable])

  // Escape cancels — a modal that traps you is worse than one you can leave.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  // --- selection ----------------------------------------------------------
  const axes = graph?.axes ?? []

  const value: VariantSelectionValue = useMemo(() => {
    const byColumn = new Map(axes.map(a => [a.column, a]))

    /**
     * Drop the "current results" flag as soon as the user chooses anything.
     *
     * A variant opens on `{CodeIsLatest: true}` — a per-row flag, NOT an axis.
     * Leaving it in place while adding `Code:f = v1` asks for rows that are
     * simultaneously the newest and the old version, which is nothing at all:
     * the figure empties and the control looks broken. An explicit choice
     * supersedes the shortcut, which is also what the user means by making it.
     */
    const withoutLatestFlag = (current: Record<string, unknown>) => {
      const flag = graph?.latest_column
      if (!flag || !(flag in current)) return current
      const next = { ...current }
      delete next[flag]
      return next
    }

    const axisForParameter = (label: string, functionLabels: string[]) => {
      const candidates = axes.filter(a => a.kind === 'param' && a.param === label)
      if (candidates.length <= 1) return candidates[0] ?? null
      // Two producing functions use this parameter name; the edge tells us
      // which one this node actually feeds.
      return (
        candidates.find(a => a.function && functionLabels.includes(a.function)) ?? null
      )
    }

    return {
      selection,
      axisForParameter,
      axisForFunction: (label: string) =>
        axes.find(a => a.kind === 'code' && a.function === label) ?? null,
      versionsFor: (label: string) => graph?.versions?.[label] ?? [],
      isLevelSelected: (column, level) => {
        const current = selection[column]
        // An axis nobody has touched selects everything: an untouched control
        // must never quietly filter data away.
        if (current === undefined) return true
        if (Array.isArray(current)) return current.map(String).includes(level)
        return String(current) === level
      },
      versionFor: column => {
        const current = selection[column]
        if (current === undefined) return 'latest'
        return Array.isArray(current) ? String(current[0] ?? 'latest') : String(current)
      },
      toggleLevel: (column, level) => {
        setSelection(previous => {
          const prev = withoutLatestFlag(previous)
          const axis = byColumn.get(column)
          const all = axis?.levels ?? []
          const current = prev[column]
          const chosen = current === undefined
            ? [...all]
            : Array.isArray(current)
              ? current.map(String)
              : [String(current)]
          const next = chosen.includes(level)
            ? chosen.filter(l => l !== level)
            : [...chosen, level]
          // Declared level order, not click order, so the legend stays stable.
          const ordered = all.filter(l => next.includes(l))
          const updated = { ...prev }
          if (ordered.length === all.length) {
            // Everything selected is the same as no constraint — and saying it
            // that way keeps the auto label ("all variants") honest.
            delete updated[column]
          } else {
            updated[column] = ordered
          }
          return updated
        })
      },
      setVersion: (column, version) => {
        setSelection(prev => ({ ...withoutLatestFlag(prev), [column]: version }))
      },
    }
  }, [axes, graph, selection])

  // Axes with no node on this canvas (produced inside a nested pipeline, say)
  // would otherwise be unreachable — the popup opens at the root scope.
  const unmapped = useMemo(() => {
    const labels = new Set(
      nodes.map(n => (n.data as { label?: string }).label ?? '')
    )
    return axes.filter(axis =>
      axis.kind === 'code'
        ? !labels.has(axis.function ?? '')
        : !labels.has(axis.param ?? '')
    )
  }, [axes, nodes])

  return (
    <div style={styles.backdrop} onClick={onCancel}>
      <div style={styles.dialog} onClick={e => e.stopPropagation()}>
        <div style={styles.header}>
          <div style={styles.headerLeft}>
            <span style={styles.title}>Select variant</span>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder={placeholder || 'variant name'}
              style={styles.nameInput}
              title="What this variant is called in the figure"
            />
          </div>
          <span style={styles.subtitle}>
            Checkboxes and versions here choose what the FIGURE shows. Nothing
            on this graph changes what a run does.
          </span>
        </div>

        <div style={styles.canvas}>
          {loading && <div style={styles.note}>Loading the pipeline…</div>}
          {error && <div style={styles.error}>Could not load the pipeline: {error}</div>}
          {!loading && !error && (
            <VariantSelectionProvider value={value}>
              <ReactFlowProvider>
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  nodeTypes={nodeTypes}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  // MUST stay true. React Flow gives a node wrapper
                  // `pointer-events: none` unless it is selectable, draggable,
                  // or carries a mouse handler — so turning all three off made
                  // every control in this popup unclickable, checkboxes as well
                  // as the version dropdowns. Selection is inert here anyway
                  // (nothing reads node.selected); it exists to keep the nodes
                  // reachable by the mouse.
                  elementsSelectable
                  fitView
                  fitViewOptions={{ padding: 0.2 }}
                  proOptions={{ hideAttribution: true }}
                >
                  <Background />
                  <Controls showInteractive={false} />
                </ReactFlow>
              </ReactFlowProvider>
            </VariantSelectionProvider>
          )}
        </div>

        {unmapped.length > 0 && (
          <div style={styles.unmapped}>
            <div style={styles.unmappedTitle}>
              Not on this canvas (defined in a nested pipeline)
            </div>
            {unmapped.map(axis => (
              <div key={axis.column} style={styles.unmappedRow}>
                <span style={styles.unmappedName}>{axis.column}</span>
                <div style={styles.unmappedLevels}>
                  {axis.levels.map(level => (
                    <label key={level} style={styles.unmappedLevel}>
                      <input
                        type="checkbox"
                        checked={value.isLevelSelected(axis.column, level)}
                        onChange={() => value.toggleLevel(axis.column, level)}
                      />
                      <span>{level}</span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={styles.footer}>
          <button type="button" style={styles.button} onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            style={styles.primaryButton}
            onClick={() => onApply({ selection, name })}
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.72)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1100,
  },
  dialog: {
    width: '82vw', height: '80vh', background: '#16162a',
    border: '1px solid #7b68ee', borderRadius: 8,
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
  },
  header: {
    padding: '10px 14px', borderBottom: '1px solid #2a2a4a', background: '#1a1a2e',
    display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  title: { color: '#eee', fontSize: 14, fontWeight: 600 },
  subtitle: { color: '#8a8aa8', fontSize: 11, fontStyle: 'italic' },
  nameInput: {
    background: '#22223a', color: '#ddd', border: '1px solid #3a3a5a',
    borderRadius: 4, fontSize: 12, padding: '3px 6px', minWidth: 200,
  },
  canvas: { flex: 1, minHeight: 0, position: 'relative' },
  footer: {
    display: 'flex', justifyContent: 'flex-end', gap: 8,
    padding: '10px 14px', borderTop: '1px solid #2a2a4a', flexShrink: 0,
  },
  button: {
    padding: '5px 14px', background: '#22223a', color: '#ccc',
    border: '1px solid #3a3a5a', borderRadius: 4, cursor: 'pointer', fontSize: 12,
  },
  primaryButton: {
    padding: '5px 14px', background: '#7b68ee', color: '#fff',
    border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12, fontWeight: 600,
  },
  unmapped: {
    borderTop: '1px solid #2a2a4a', padding: '8px 14px', maxHeight: 120,
    overflowY: 'auto', flexShrink: 0,
  },
  unmappedTitle: { fontSize: 10, color: '#e0b050', marginBottom: 4 },
  unmappedRow: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 3 },
  unmappedName: { fontSize: 11, fontFamily: 'monospace', color: '#bbb' },
  unmappedLevels: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  unmappedLevel: {
    display: 'flex', alignItems: 'center', gap: 3, fontSize: 11, color: '#ddd',
  },
  note: { fontSize: 12, color: '#777', fontStyle: 'italic', padding: 12 },
  error: { fontSize: 12, color: '#f87171', padding: 12 },
  inertNode: {
    background: '#2a2438', border: '2px dashed #6b5a9a', borderRadius: 6,
    padding: '8px 12px', minWidth: 150, opacity: 0.55,
  },
  inertNodeLabel: {
    fontWeight: 600, color: '#c4b5fd', fontFamily: 'monospace',
    textAlign: 'center', fontSize: 13,
  },
  inertNodeHint: {
    fontSize: 10, color: '#8a8aa8', fontStyle: 'italic', textAlign: 'center',
  },
}
