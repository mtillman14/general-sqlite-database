/**
 * Root application component.
 *
 * Layout:
 *   ┌─────────────────────────────────────────────┐
 *   │  header: SciStack + db name                 │
 *   ├───────────────────────────────┬─────────────┤
 *   │  PipelineDAG (left 3/4)       │  sidebar    │
 *   │                               │  (right 1/4)│
 *   └───────────────────────────────┴─────────────┘
 */

import { useEffect, useState, useCallback } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import PipelineDAG from './components/DAG/PipelineDAG'
import Sidebar from './components/Sidebar/Sidebar'
import { RunLogProvider } from './context/RunLogContext'
import { SelectedNodeProvider } from './context/SelectedNodeContext'

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    height: '100%',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 18px',
    background: '#1a1a2e',
    color: '#fff',
    fontSize: 14,
    flexShrink: 0,
  },
  title: {
    fontWeight: 700,
    fontSize: 16,
    letterSpacing: 0.5,
  },
  separator: {
    opacity: 0.4,
  },
  dbName: {
    fontFamily: 'monospace',
    opacity: 0.8,
  },
  refreshBtn: {
    marginLeft: 'auto',
    padding: '4px 12px',
    background: '#2a2a4a',
    color: '#ccc',
    border: '1px solid #3a3a5a',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: 'inherit',
  },
  schemaKeys: {
    opacity: 0.6,
    fontSize: 12,
  },
  body: {
    display: 'flex',
    flexDirection: 'row',
    flex: 1,
    minHeight: 0,
  },
  dagArea: {
    flex: 3,
    minWidth: 0,
    minHeight: 0,
  },
  sidebar: {
    flex: 1,
    minWidth: 0,
    borderLeft: '1px solid #2a2a4a',
    background: '#12122a',
  },
}

export default function App() {
  const [schema, setSchema] = useState<{ keys: string[] }>({ keys: [] })
  const [dbName, setDbName] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  const handleRefresh = useCallback(async () => {
    setRefreshing(true)
    try {
      const res = await fetch('/api/refresh', { method: 'POST' })
      const data = await res.json()
      if (!data.ok) {
        console.error('Refresh failed:', data.error)
      }
    } catch (err) {
      console.error('Refresh failed:', err)
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetch('/api/schema')
      .then((r) => r.json())
      .then(setSchema)
      .catch(console.error)
    fetch('/api/info')
      .then((r) => r.json())
      .then((d) => setDbName(d.db_name))
      .catch(console.error)
  }, [])

  return (
    <RunLogProvider>
    <SelectedNodeProvider>
    <div style={styles.root}>
      <header style={styles.header}>
        <span style={styles.title}>SciStack</span>
        <span style={styles.separator}>|</span>
        <span style={styles.dbName}>{dbName || 'loading…'}</span>
        <button
          style={styles.refreshBtn}
          onClick={handleRefresh}
          disabled={refreshing}
          title="Re-import the pipeline module from disk"
        >
          {refreshing ? 'Refreshing...' : 'Refresh Module'}
        </button>
        {schema.keys.length > 0 && (
          <span style={styles.schemaKeys}>
            schema: [{schema.keys.join(', ')}]
          </span>
        )}
      </header>
      <ReactFlowProvider>
        <div style={styles.body}>
          <div style={styles.dagArea}>
            <PipelineDAG />
          </div>
          <div style={styles.sidebar}>
            <Sidebar />
          </div>
        </div>
      </ReactFlowProvider>
    </div>
    </SelectedNodeProvider>
    </RunLogProvider>
  )
}
