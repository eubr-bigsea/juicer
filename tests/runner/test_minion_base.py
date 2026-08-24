# -*- coding: utf-8 -*-
import queue

from juicer.runner.minion_base import Minion


class _FlakyMinion(Minion):
    """A minion whose heartbeat fails a couple of times before succeeding,
    to prove ping() survives a transient error instead of dying silently."""

    def __init__(self, stop_queue):
        self.pid = 1
        self.calls = 0
        self._stop_queue = stop_queue

    def _perform_ping(self):
        self.calls += 1
        if self.calls < 3:
            raise ConnectionError('redis hiccup')
        self._stop_queue.put(1)  # stop the loop on the next check


def test_ping_survives_transient_error(monkeypatch):
    monkeypatch.setattr('juicer.runner.minion_base.time.sleep', lambda s: None)

    q = queue.Queue()
    minion = _FlakyMinion(q)
    minion.ping(q)

    assert minion.calls == 3
