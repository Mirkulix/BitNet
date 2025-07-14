"""Configuration management."""

from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
