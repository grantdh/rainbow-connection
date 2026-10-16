"""events.py: ring buffer replay, event bus seq/replay/heartbeat/terminal close, SSE framing."""

from __future__ import annotations

import json
import threading
import unittest

from abrun import events


class TestRingBuffer(unittest.TestCase):
    def test_since_and_tail(self):
        rb = events.RingBuffer(maxlen=5)
        for i in range(7):
            rb.append("out" if i % 2 else "err", f"line {i}")
        lines, nxt = rb.since(0)
        self.assertEqual([l[2] for l in lines], ["line 2", "line 3", "line 4", "line 5", "line 6"])   # ring dropped 0, 1
        self.assertEqual(nxt, 7)
        lines, nxt = rb.since(7)
        self.assertEqual((lines, nxt), ([], 7))
        lines, _ = rb.since(0, stream="out")
        self.assertEqual([l[2] for l in lines], ["line 3", "line 5"])
        lines, nxt = rb.since(0, max_lines=2)
        self.assertEqual(([l[0] for l in lines], nxt), ([3, 4], 4))
        self.assertEqual(rb.tail(2), ["line 5", "line 6"])
        self.assertTrue(rb.wait(0, 0.01))
        self.assertFalse(rb.wait(7, 0.01))


class TestEventBus(unittest.TestCase):
    def test_replay_and_terminal(self):
        bus = events.EventBus()
        bus.publish("execution_start", {"run_id": "r"})
        bus.publish("muppet.log", {"run_id": "r", "text": "x"})
        self.assertEqual([e[2] for e in bus.since(0)], ["execution_start", "muppet.log"])
        self.assertEqual([e[2] for e in bus.since(1)], ["muppet.log"])
        got = []
        t = threading.Thread(target=lambda: got.extend(bus.subscribe(since=1, heartbeat_s=0.05, close_on_terminal=True)), daemon=True)
        t.start()
        bus.publish("muppet.done", {"run_id": "r", "status": "success"})
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual([g[1] for g in got if g[1] != "heartbeat"], ["muppet.log", "muppet.done"])
        self.assertTrue(bus.closed)

    def test_heartbeat_and_stop(self):
        bus = events.EventBus()
        stop = threading.Event()
        got = []

        def consume():
            for item in bus.subscribe(since=0, stop=stop, heartbeat_s=0.05):
                got.append(item)
                if len(got) >= 2:
                    stop.set()
        t = threading.Thread(target=consume, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual([g[1] for g in got], ["heartbeat", "heartbeat"])

    def test_sse_frames(self):
        self.assertEqual(events.sse_frame(None, "heartbeat", {}), b": ping\n\n")
        frame = events.sse_frame(7, "progress", {"run_id": "r", "value": 3, "max": 100})
        lines = frame.decode().split("\n")
        self.assertEqual(lines[0], "id: 7")
        self.assertTrue(lines[1].startswith("data: "))
        self.assertEqual(json.loads(lines[1][6:]), {"type": "progress", "data": {"run_id": "r", "value": 3, "max": 100}})
        self.assertEqual(lines[2:], ["", ""])
        bus = events.EventBus()
        bus.publish("status", {"queue_remaining": 0})
        bus.publish("muppet.done", {"run_id": "r"})
        frames = list(events.sse_stream(bus, since=0, close_on_terminal=True))
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].startswith(b"id: 1\n"))


if __name__ == "__main__":
    unittest.main()
