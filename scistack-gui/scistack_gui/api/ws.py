"""
WebSocket endpoint: /ws

A WebSocket is a persistent two-way connection between the browser and the
server — unlike HTTP where the browser asks and the server answers once,
a WebSocket stays open so the server can push messages at any time.

We use it here to stream for_each stdout back to the frontend in real time,
and to notify the frontend when the DAG should refresh.

Messages sent to the frontend are JSON objects with a "type" field:
  {"type": "run_output", "run_id": "...", "text": "..."}
  {"type": "run_done",   "run_id": "...", "success": true}
  {"type": "dag_updated"}
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# All currently connected browser clients. We broadcast to all of them.
_clients: list[WebSocket] = []

# asyncio queue for messages from background threads to WebSocket clients.
_queue: asyncio.Queue | None = None

# The running event loop — stored from async context so background threads
# can schedule work on it via call_soon_threadsafe.
# (asyncio.get_event_loop() raises RuntimeError in non-main threads in Python 3.10+)
_loop: asyncio.AbstractEventLoop | None = None


def get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


def push_message(msg: dict) -> None:
    """
    Thread-safe: called from a background thread to enqueue a message.
    Uses call_soon_threadsafe so the put is scheduled on the event loop
    thread rather than called directly from the background thread.
    """
    if _loop is None:
        return   # No WebSocket client connected yet; drop the message
    _loop.call_soon_threadsafe(get_queue().put_nowait, msg)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _loop
    _loop = asyncio.get_running_loop()   # capture from async context
    await websocket.accept()
    _clients.append(websocket)
    try:
        # Pump messages from the queue to all connected clients.
        # We also need to keep listening for client messages (e.g. ping).
        # Run both concurrently with asyncio.gather.
        await asyncio.gather(
            _pump_queue(websocket),
            _listen(websocket),
        )
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _clients:
            _clients.remove(websocket)


async def _pump_queue(websocket: WebSocket):
    """Forward messages from the shared queue to this client."""
    q = get_queue()
    while True:
        msg = await q.get()
        # Broadcast to all connected clients
        for client in list(_clients):
            try:
                await client.send_json(msg)
            except Exception:
                pass


async def _listen(websocket: WebSocket):
    """Keep the connection alive by consuming incoming messages."""
    while True:
        await websocket.receive_text()


async def broadcast(msg: dict) -> None:
    """Send a message to all connected clients from async context."""
    for client in list(_clients):
        try:
            await client.send_json(msg)
        except Exception:
            pass
