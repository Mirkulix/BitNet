"""Minimal SNN module placeholder."""


class SpikingNeuron:
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.potential = 0.0

    def step(self, input_current: float) -> bool:
        self.potential += input_current
        if self.potential >= self.threshold:
            self.potential = 0.0
            return True
        return False
