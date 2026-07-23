from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from jumpKingConjunto import jumpKingConjunto

print("NIVELES 4")

training_levels = [4]

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
    save_path="./checkpoint4.1",
    name_prefix="conjunto4.1"
)

model.learn(
    total_timesteps=200_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save("jumpKingConj4.1")