"""Evaluation metrics."""


def accuracy(preds, labels) -> float:
    correct = sum(p == label for p, label in zip(preds, labels))
    return correct / len(labels)
