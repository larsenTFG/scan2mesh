"""Progress reporting utilities."""

from __future__ import annotations

from typing import Callable, Optional


class ProgressReporter:
    """Wraps a progress callback with stage-based percentage tracking."""

    # Pipeline stages and their weight in the overall progress
    STAGES = {
        "read": (0.0, 0.10),
        "preprocess": (0.10, 0.25),
        "normals": (0.25, 0.40),
        "reconstruct": (0.40, 0.70),
        "postprocess": (0.70, 0.90),
        "export": (0.90, 1.0),
    }

    def __init__(self, callback: Optional[Callable[[str, float], None]] = None):
        self.callback = callback

    def report(self, stage: str, message: str, stage_progress: float = 0.0) -> None:
        """Report progress within a pipeline stage.

        Args:
            stage: One of the STAGES keys.
            message: Human-readable status message.
            stage_progress: Progress within this stage (0.0 to 1.0).
        """
        if self.callback is None:
            return
        start, end = self.STAGES.get(stage, (0.0, 1.0))
        overall = start + (end - start) * min(stage_progress, 1.0)
        self.callback(message, overall)
