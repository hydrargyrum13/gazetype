from __future__ import annotations


class DeliberateBlinkDetector:
    def __init__(
        self,
        close_threshold: float = 0.58,
        open_threshold: float = 0.34,
        minimum_ms: int = 250,
        maximum_ms: int = 800,
        refractory_ms: int = 900,
    ):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.minimum_ms = minimum_ms
        self.maximum_ms = maximum_ms
        self.refractory_ms = refractory_ms
        self._closed_since: int | None = None
        self._armed_when_closed = False
        self._last_triggered = -refractory_ms

    def reset(self) -> None:
        self._closed_since = None
        self._armed_when_closed = False

    def update(self, timestamp_ms: int, left: float, right: float, armed: bool) -> bool:
        both_closed = left >= self.close_threshold and right >= self.close_threshold
        both_open = left <= self.open_threshold and right <= self.open_threshold

        if both_closed and self._closed_since is None:
            self._closed_since = timestamp_ms
            self._armed_when_closed = armed
            return False

        if self._closed_since is None:
            return False

        duration = timestamp_ms - self._closed_since
        if duration > self.maximum_ms:
            self.reset()
            return False

        if both_open:
            valid = (
                self._armed_when_closed
                and armed
                and self.minimum_ms <= duration <= self.maximum_ms
                and timestamp_ms - self._last_triggered >= self.refractory_ms
            )
            self.reset()
            if valid:
                self._last_triggered = timestamp_ms
                return True
        return False

