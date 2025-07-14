from src.training.rl_trainer import RLTrainer

trainer = RLTrainer(model=None)
print("step result:", trainer.step(batch=None))
