"""Temporal data utilities."""

from typing import Sequence


def sliding_window(sequence: Sequence, window: int):
    for i in range(len(sequence) - window + 1):
        yield sequence[i : i + window]
