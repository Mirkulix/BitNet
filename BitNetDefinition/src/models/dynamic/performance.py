"""Performance monitoring utils."""

from typing import List


class PerformanceTracker:
    def __init__(self):
        self.metrics: List[float] = []

    def update(self, value: float):
        self.metrics.append(value)

    def mean(self) -> float:
        return sum(self.metrics) / len(self.metrics) if self.metrics else 0.0
