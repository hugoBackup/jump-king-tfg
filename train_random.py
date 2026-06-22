from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from jumpKingRandomScreen import JumpKingRandomScreen

env = JumpKingRandomScreen()

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
    save_path="./checkpointRandomDesde0",
    name_prefix="randomScreen_"
)

model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save(
    "jumpKingRandomDesde0"
)