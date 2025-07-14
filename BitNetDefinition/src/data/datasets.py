"""Dataset placeholders."""


class DummyDataset:
    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx
