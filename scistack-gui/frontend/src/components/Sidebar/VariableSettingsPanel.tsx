/**
 * VariableSettingsPanel — shown in the sidebar when a variable node is selected.
 *
 * Fetches GET /api/variables/{label}/records and displays:
 *   1. Variant summary: one row per unique branch_params combination + record count.
 *   2. Records table: one row per record with schema key values and variant label.
 */

import { useEffect, useState } from 'react'

interface VariantSummary {
  label: string
  branch_params: Record<string, unknown>
  record_count: number
}

interface RecordRow {
  [key: string]: string | null
  variant_label: string
}

interface VariableRecordsResponse {
  schema_keys: string[]
  records: RecordRow[]
  variants: VariantSummary[]
}

interface Props {
  label: string
}

export default function VariableSettingsPanel({ label }: Props) {
  const [data, setData] = useState<VariableRecordsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setData(null)
    setError(null)
    fetch(`/api/variables/${encodeURIComponent(label)}/records`)
      .then(r => {
        if (!r.ok) return r.json().then(e => Promise.reject(e.detail ?? 'Error'))
        return r.json()
      })
      .then(setData)
      .catch(err => setError(String(err)))
  }, [label])

  return (
    <div style={styles.root}>
      <div style={styles.varName}>{label}</div>

      {error && <div style={styles.error}>{error}</div>}
      {!data && !error && <div style={styles.loading}>Loading…</div>}

      {data && (
        <>
          {/* Variant summary */}
          <section style={styles.section}>
            <div style={styles.sectionTitle}>Variants</div>
            {data.variants.length === 0 ? (
              <div style={styles.empty}>No records.</div>
            ) : (
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Variant</th>
                    <th style={{ ...styles.th, textAlign: 'right' }}>Records</th>
                  </tr>
                </thead>
                <tbody>
                  {data.variants.map((v, i) => (
                    <tr key={i} style={styles.row}>
                      <td style={styles.td}>
                        <span style={styles.pill}>{v.label}</span>
                      </td>
                      <td style={{ ...styles.td, textAlign: 'right', color: '#888' }}>
                        {v.record_count}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* Records table */}
          {data.records.length > 0 && (
            <section style={styles.section}>
              <div style={styles.sectionTitle}>Records</div>
              <table style={styles.table}>
                <thead>
                  <tr>
                    {data.schema_keys.map(k => (
                      <th key={k} style={styles.th}>{k}</th>
                    ))}
                    {data.variants.length > 1 && (
                      <th style={styles.th}>variant</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.records.map((rec, i) => (
                    <tr key={i} style={styles.row}>
                      {data.schema_keys.map(k => (
                        <td key={k} style={styles.td}>
                          <span style={styles.pill}>{rec[k] ?? '—'}</span>
                        </td>
                      ))}
                      {data.variants.length > 1 && (
                        <td style={styles.td}>
                          <span style={{ ...styles.pill, color: '#a89cf0' }}>
                            {rec.variant_label}
                          </span>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    padding: '12px',
    color: '#ccc',
    fontSize: 12,
  },
  varName: {
    fontFamily: 'monospace',
    fontWeight: 700,
    fontSize: 13,
    color: '#4a90d9',
    marginBottom: 12,
    wordBreak: 'break-all',
  },
  loading: {
    color: '#555',
    fontStyle: 'italic',
    fontSize: 11,
  },
  error: {
    color: '#e07070',
    fontSize: 11,
    fontStyle: 'italic',
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    color: '#666',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  empty: {
    color: '#555',
    fontStyle: 'italic',
    fontSize: 11,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  th: {
    textAlign: 'left',
    fontSize: 10,
    color: '#888',
    fontWeight: 600,
    padding: '2px 4px 4px 0',
    borderBottom: '1px solid #2a2a4a',
    fontFamily: 'monospace',
  },
  row: {
    borderBottom: '1px solid #1e1e3a',
  },
  td: {
    padding: '4px 4px 4px 0',
    verticalAlign: 'middle',
  },
  pill: {
    display: 'inline-block',
    background: '#1e1e3a',
    borderRadius: 3,
    padding: '1px 5px',
    fontFamily: 'monospace',
    fontSize: 11,
    color: '#b2ded9',
  },
}
