import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


from envs.jumpKingBaseEnv import JumpKingBaseEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay



os.environ["render"] = "1"

env = jumpKingAgentCurvedRay()

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
    save_path="./checkpointCurvedRay",
    name_prefix="curvedRay"
)

model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

model.save(
    "jumpKingCurvedRay"
)