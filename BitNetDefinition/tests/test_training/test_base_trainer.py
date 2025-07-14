from src.training.base_trainer import BaseTrainer


def test_base_trainer_init():
    trainer = BaseTrainer(model=None)
    assert trainer.model is None
