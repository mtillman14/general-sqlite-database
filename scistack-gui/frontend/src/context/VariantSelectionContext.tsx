/**
 * VariantSelectionContext — what a DAG node's controls mean *right now*.
 *
 * The pipeline canvas and Plot Studio's variant popup draw the same nodes with
 * the same widgets, and they mean two completely different things:
 *
 *   - On the canvas, a ParameterNode checkbox is EXECUTION state. Unchecking a
 *     value excludes it from future for_each fan-outs (pipeline_store's
 *     hide_constant_value, and execution_service's filtering).
 *   - In the popup, the same checkbox is DISPLAY state. It selects which
 *     already-computed records a figure draws, and must never change what a run
 *     would do.
 *
 * Binding the popup to the canvas's handlers would make looking at a plot
 * quietly rewrite the run configuration — far worse than any duplicated widget.
 * So the mode is carried here rather than in node data: when this context is
 * present a node is in *selection* mode and the execution calls are unreachable
 * from it; when it is absent (the canvas, always) nothing about today's
 * behaviour changes.
 *
 * See docs/claude/variant-selection.md §5 and .claude/plan-plot-variant-rows.md.
 */

import { createContext, useContext } from 'react'

/** One variant axis, as `plot_variant_graph` describes it. */
export interface VariantAxis {
  column: string
  kind: 'code' | 'param'
  function: string | null
  param: string | null
  levels: string[]
}

export interface FunctionVersion {
  version: string
  function_hash: string
  first_saved: string | null
}

export interface VariantSelectionValue {
  /** The selection being edited: `{column: level | level[] | 'latest'}`. */
  selection: Record<string, unknown>
  /** Axes of the plotted measure, keyed by the node they belong to. */
  axisForParameter: (parameterLabel: string, functionLabels: string[]) => VariantAxis | null
  axisForFunction: (functionLabel: string) => VariantAxis | null
  /** Every recorded version of a function, newest last. Empty = never run. */
  versionsFor: (functionLabel: string) => FunctionVersion[]
  /** Check/uncheck one level of a parameter axis. */
  toggleLevel: (column: string, level: string) => void
  /** Pick a version (or 'latest') for a function axis. */
  setVersion: (column: string, version: string) => void
  /** True when a level is currently selected. An axis with nothing chosen means
   *  "every level" — an unedited axis must not filter anything away. */
  isLevelSelected: (column: string, level: string) => boolean
  /** The version chosen for a function, defaulting to 'latest'. */
  versionFor: (column: string) => string
}

const VariantSelectionContext = createContext<VariantSelectionValue | null>(null)

export const VariantSelectionProvider = VariantSelectionContext.Provider

/** Non-null only inside the variant popup. Nodes branch on it. */
export function useVariantSelection(): VariantSelectionValue | null {
  return useContext(VariantSelectionContext)
}
