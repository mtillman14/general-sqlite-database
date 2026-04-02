/**
 * VariableNode — represents a named variable type in the pipeline.
 *
 * Shows the variable type name and a scrollable checkboxed list of its
 * variants. Checked variants are "selected" for downstream runs.
 *
 * State management: checked state lives inside each variant object in the
 * node's `data`. We update it via useReactFlow().setNodes — React Flow's
 * own hook for mutating the node graph from inside a custom node. This
 * avoids needing to pass callbacks through props.
 */

import { useCallback } from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'

export interface Variant {
  constants: Record<string, unknown>
  record_count: number
  checked: boolean
}

export interface VariableNodeData {
  label: string
  variants: Variant[]
  total_records: number
}

interface Props {
  id: string           // React Flow passes the node's id automatically
  data: VariableNodeData
}

export default function VariableNode({ id, data }: Props) {
  const { setNodes } = useReactFlow()

  const toggleVariant = useCallback((index: number) => {
    setNodes(nds => nds.map(node => {
      if (node.id !== id) return node
      const variants = (node.data.variants as Variant[]).map((v, i) =>
        i === index ? { ...v, checked: !v.checked } : v
      )
      return { ...node, data: { ...node.data, variants } }
    }))
  }, [id, setNodes])

  return (
    <div style={styles.container}>
      <Handle type="target" position={Position.Left} />

      <div style={styles.label}>{data.label}</div>

      <div style={styles.listbox}>
        {data.variants.length === 0 ? (
          <span style={data.total_records > 0 ? styles.rawCount : styles.empty}>
            {data.total_records > 0
              ? `${data.total_records} record${data.total_records !== 1 ? 's' : ''}`
              : 'empty'}
          </span>
        ) : (
          data.variants.map((v, i) => {
            const constantsLabel = Object.entries(v.constants)
              .map(([k, val]) => `${k}=${val}`)
              .join(', ')
            const countLabel = `${v.record_count} record${v.record_count !== 1 ? 's' : ''}`
            const rowLabel = constantsLabel
              ? `${constantsLabel} · ${countLabel}`
              : countLabel
            const showCheckbox = data.variants.length > 1

            return (
              <label key={i} style={showCheckbox ? styles.variantRow : styles.variantRowNoCheck}>
                {showCheckbox && (
                  <input
                    type="checkbox"
                    checked={v.checked}
                    onChange={() => toggleVariant(i)}
                    style={styles.checkbox}
                  />
                )}
                <span style={!showCheckbox || v.checked ? styles.variantLabel : styles.variantLabelUnchecked}>
                  {rowLabel}
                </span>
              </label>
            )
          })
        )}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: '#fff',
    border: '2px solid #4a90d9',
    borderRadius: 8,
    padding: '8px 12px',
    minWidth: 180,
    fontSize: 13,
    boxShadow: '0 2px 6px rgba(0,0,0,0.12)',
  },
  label: {
    fontWeight: 600,
    color: '#1a1a2e',
    marginBottom: 5,
  },
  listbox: {
    maxHeight: 110,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
  },
  variantRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    cursor: 'pointer',
    userSelect: 'none',
  },
  variantRowNoCheck: {
    display: 'flex',
    alignItems: 'center',
    userSelect: 'none',
  },
  checkbox: {
    margin: 0,
    cursor: 'pointer',
    accentColor: '#4a90d9',
    flexShrink: 0,
  },
  variantLabel: {
    fontSize: 11,
    color: '#333',
  },
  variantLabelUnchecked: {
    fontSize: 11,
    color: '#aaa',
    textDecoration: 'line-through',
  },
  rawCount: {
    fontSize: 11,
    color: '#666',
  },
  empty: {
    fontSize: 11,
    color: '#bbb',
    fontStyle: 'italic',
  },
}
