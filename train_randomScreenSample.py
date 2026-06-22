from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from jumpKingRandomScreenSample import JumpKingRandomScreenSample

env = JumpKingRandomScreenSample()

model = PPO.load(
    "checkpointRandomSample/randomScreen__82000_steps",
    env=env
)

checkpoint_callback = CheckpointCallback(
    save_freq=1_000,
    save_path="./checkpointRandomSample",
    name_prefix="randomScreen_"
)

model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save(
    "jumpKingRandomSample"
)