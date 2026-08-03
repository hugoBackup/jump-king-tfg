import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.jumpKingStraightRayEnv import JumpKingStraightRayEnv

os.environ["render"] = "0"

env = JumpKingStraightRayEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpointStraightRay100DQN",
    name_prefix="straightRay100DQN"
)

MAX_TIME = 4 * 24 * 60 * 60  # 4 días

start_time = time.time()

while time.time() - start_time < MAX_TIME:

    model.learn(
        total_timesteps=10_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

model.save("jumpKingStraightRay100DQN_4days")