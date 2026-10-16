"""Ring buffers and the event bus (per run and global), plus the SSE framing the server will stream.

Events are {type, data} envelopes with the studio's names (status, execution_start, executing, progress,
muppet.log, executed, execution_error, execution_interrupted, muppet.done) plus abrun.state.  Every
published event carries a monotonically increasing seq so a reconnecting EventSource (Last-Event-ID or
?since=) replays what it missed.  Log lines live in a separate 20,000-line ring per run.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

EVENT_TYPES = ("status", "execution_start", "executing", "progress", "muppet.log", "executed", "execution_error",
               "execution_interrupted", "muppet.done", "abrun.state")
TERMINAL_EVENT = "muppet.done"


class RingBuffer:
    """seq-numbered ring of (seq, stream, text) log lines."""

    def __init__(self, maxlen: int = 20000):
        self._buf: deque = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def append(self, stream: str, text: str) -> int:
        with self._cv:
            self._seq += 1
            self._buf.append((self._seq, stream, text))
            self._cv.notify_all()
            return self._seq

    @property
    def seq(self) -> int:
        return self._seq

    def since(self, seq: int = 0, max_lines: int = 2000, stream: str | None = None) -> tuple[list, int]:
        """(lines after seq, next seq to ask for).  stream filters 'out'|'err'|'guard'; None = both."""
        with self._lock:
            items = [it for it in self._buf if it[0] > seq and (stream in (None, "both") or it[1] == stream)]
        items = items[:max_lines]
        nxt = items[-1][0] if items else seq
        return [list(it) for it in items], nxt

    def tail(self, n: int = 25, stream: str | None = None) -> list[str]:
        with self._lock:
            items = [it for it in self._buf if stream in (None, "both") or it[1] == stream]
        return [it[2] for it in items[-n:]]

    def wait(self, seq: int, timeout: float) -> bool:
        with self._cv:
            if self._seq > seq:
                return True
            return self._cv.wait(timeout)


class EventBus:
    """seq-numbered event history with blocking subscribers; one per run and one global."""

    def __init__(self, maxlen: int = 5000):
        self._events: deque = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.closed = False

    def publish(self, etype: str, data: dict | None = None) -> int:
        with self._cv:
            self._seq += 1
            self._events.append((self._seq, time.time(), etype, data or {}))
            if etype == TERMINAL_EVENT:
                self.closed = True
            self._cv.notify_all()
            return self._seq

    @property
    def seq(self) -> int:
        return self._seq

    def since(self, seq: int = 0) -> list[tuple]:
        with self._lock:
            return [e for e in self._events if e[0] > seq]

    def wait(self, seq: int, timeout: float) -> bool:
        with self._cv:
            if self._seq > seq:
                return True
            return self._cv.wait(timeout)

    def subscribe(self, since: int = 0, stop=None, heartbeat_s: float = 10.0, close_on_terminal: bool = False):
        """Generator of (seq, etype, data) tuples, or (None, 'heartbeat', {}) every heartbeat_s of silence.
        Ends when stop (a threading.Event) is set, or after the terminal event when close_on_terminal."""
        seq = since
        while not (stop is not None and stop.is_set()):
            events = self.since(seq)
            if not events:
                if close_on_terminal and self.closed:
                    return
                if not self.wait(seq, heartbeat_s):
                    yield None, "heartbeat", {}
                continue
            for s, _t, etype, data in events:
                seq = s
                yield s, etype, data
                if close_on_terminal and etype == TERMINAL_EVENT:
                    return


def sse_frame(seq: int | None, etype: str, data: dict) -> bytes:
    """One Server-Sent-Events frame: 'id: <seq>' + 'data: {"type":..,"data":..}' (heartbeat -> ': ping')."""
    if seq is None:
        return b": ping\n\n"
    payload = json.dumps({"type": etype, "data": data}, default=str)
    return f"id: {seq}\ndata: {payload}\n\n".encode()


def sse_stream(bus: EventBus, since: int = 0, stop=None, heartbeat_s: float = 10.0, close_on_terminal: bool = False):
    """Byte frames for an HTTP handler to write (the server agent's SSE endpoints call this)."""
    for seq, etype, data in bus.subscribe(since, stop, heartbeat_s, close_on_terminal):
        yield sse_frame(seq, etype, data)
