
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.jumpKingBaseEnv import JumpKingBaseEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay



os.environ["render"] = "0"

from envs.JumpkingcurvedRayEnv import JumpKingCurvedRayEnv

env = JumpKingCurvedRayEnv()

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


MAX_TIME = 4 * 24 * 60 * 60

start_time = time.time()

while time.time() - start_time < MAX_TIME:

    model.learn(
        total_timesteps=10_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

model.save("jumpKingCurvedRay_4days")