"""Timestamp management for expansion events."""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Timestamp:
    event: str
    time: datetime


class TimestampManager:
    def __init__(self):
        self.events: List[Timestamp] = []

    def record(self, event: str):
        self.events.append(Timestamp(event=event, time=datetime.utcnow()))
