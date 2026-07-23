from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.jumpKingStraightRayEnv import JumpKingStraightRayEnv

MODEL_NAME = "jumpKingStraightRay"

env = JumpKingStraightRayEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpointStraightRay",
    name_prefix="straightRay"
)

model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save(MODEL_NAME)

env.close()