import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

os.environ["render"] = "0"

from envs.JumpkingcurvedRayEnv import JumpKingCurvedRayEnv

env = JumpKingCurvedRayEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpointCurvedRay100DQN",
    name_prefix="curvedRay100DQN"
)


MAX_TIME = 4 * 24 * 60 * 60

start_time = time.time()

while time.time() - start_time < MAX_TIME:

    model.learn(
        total_timesteps=10_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

model.save("jumpKingCurvedRay100DQN_4days")