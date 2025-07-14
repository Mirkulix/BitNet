"""RL trainer placeholder."""


class RLTrainer:
    def __init__(self, model, algorithm: str = "ppo"):
        self.model = model
        self.algorithm = algorithm

    def step(self, batch):  # pragma: no cover
        return 0.0
