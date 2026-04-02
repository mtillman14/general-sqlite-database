/**
 * useWebSocket — connects to the backend WebSocket and exposes incoming messages.
 *
 * The WebSocket stays open for the lifetime of the app. Components subscribe
 * by passing an onMessage callback. The hook reconnects automatically if the
 * connection drops (e.g. backend restart).
 */

import { useEffect, useRef, useCallback } from 'react'

type MessageHandler = (msg: Record<string, unknown>) => void

const WS_URL = `ws://${window.location.hostname}:8765/ws`

// Module-level singleton so all hook instances share one connection.
let _socket: WebSocket | null = null
const _handlers = new Set<MessageHandler>()

function getSocket(): WebSocket {
  if (_socket && _socket.readyState <= WebSocket.OPEN) return _socket

  _socket = new WebSocket(WS_URL)

  _socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data)
      _handlers.forEach(h => h(msg))
    } catch {
      // ignore malformed messages
    }
  }

  _socket.onclose = () => {
    // Reconnect after 2 seconds
    setTimeout(getSocket, 2000)
  }

  return _socket
}

export function useWebSocket(onMessage: MessageHandler) {
  const handlerRef = useRef(onMessage)
  handlerRef.current = onMessage

  // Stable wrapper so the Set entry never changes identity
  const stableHandler = useCallback((msg: Record<string, unknown>) => {
    handlerRef.current(msg)
  }, [])

  useEffect(() => {
    getSocket()
    _handlers.add(stableHandler)
    return () => { _handlers.delete(stableHandler) }
  }, [stableHandler])
}
