from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from gazetype.models import GazePoint, SensitivityProfile


@dataclass(slots=True)
class _Candidate:
    key_id: str
    started_ms: int
    frames: int = 1


class LandingDetector:
    """Selects only the low-velocity landing after a rapid gaze movement."""

    def __init__(self, profile: SensitivityProfile):
        self.profile = profile
        self._previous: GazePoint | None = None
        self._previous_key: str | None = None
        self._candidate: _Candidate | None = None
        self._locked_key: str | None = None
        self._saccade_anchor_ms: int | None = None

    @property
    def candidate_progress(self) -> float:
        if not self._candidate:
            return 0.0
        frame_progress = self._candidate.frames / self.profile.minimum_frames
        if not self._previous:
            return min(frame_progress, 1.0)
        elapsed = self._previous.timestamp_ms - self._candidate.started_ms
        time_progress = elapsed / self.profile.minimum_ms
        return max(0.0, min(frame_progress, time_progress, 1.0))

    @property
    def candidate_key(self) -> str | None:
        return self._candidate.key_id if self._candidate else None

    def reset(self) -> None:
        self._previous = None
        self._previous_key = None
        self._candidate = None
        self._locked_key = None
        self._saccade_anchor_ms = None

    def cancel_candidate(self) -> None:
        self._candidate = None

    def update(self, point: GazePoint, key_id: str | None) -> str | None:
        velocity = self._velocity(point)

        if self._locked_key is not None and key_id != self._locked_key:
            self._locked_key = None

        if velocity >= self.profile.saccade_velocity:
            self._saccade_anchor_ms = self._previous.timestamp_ms if self._previous else point.timestamp_ms
            self._candidate = None
            self._previous = point
            self._previous_key = key_id
            return None

        if key_id is None or velocity > self.profile.landing_velocity:
            self._candidate = None
            self._previous = point
            self._previous_key = key_id
            return None

        if self._locked_key == key_id:
            self._candidate = None
            self._previous = point
            self._previous_key = key_id
            return None

        if self._candidate is None or self._candidate.key_id != key_id:
            start = self._saccade_anchor_ms if self._saccade_anchor_ms is not None else point.timestamp_ms
            landing_frames = 2 if self._saccade_anchor_ms is not None and self._previous_key == key_id else 1
            self._candidate = _Candidate(key_id, start, landing_frames)
        else:
            self._candidate.frames += 1

        elapsed = point.timestamp_ms - self._candidate.started_ms
        selected = None
        if (
            self._candidate.frames >= self.profile.minimum_frames
            and elapsed >= self.profile.minimum_ms
        ):
            selected = key_id
            self._locked_key = key_id
            self._candidate = None
            self._saccade_anchor_ms = None

        self._previous = point
        self._previous_key = key_id
        return selected

    def _velocity(self, point: GazePoint) -> float:
        if self._previous is None:
            return 0.0
        delta_ms = max(point.timestamp_ms - self._previous.timestamp_ms, 1)
        return hypot(point.x - self._previous.x, point.y - self._previous.y) / (delta_ms / 1000.0)
