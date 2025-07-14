"""Base trainer placeholder."""


class BaseTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, data_loader):  # pragma: no cover - requires torch
        for _ in data_loader:
            pass
