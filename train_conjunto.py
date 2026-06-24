from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from jumpKingConjunto import jumpKingConjunto

print("NIVELES 20")

training_levels = [20]

env = jumpKingConjunto(
    training_levels=training_levels
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64
)

checkpoint_callback = CheckpointCallback(
    save_freq=1_000,
    save_path="./checkpoint20.1",
    name_prefix="conjunto20.1_"
)

model.learn(
    total_timesteps=200_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save("jumpKingConj20.1")