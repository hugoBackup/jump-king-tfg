import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO

from envs.jumpKingStraightRayEnv import JumpKingStraightRayEnv

os.environ["render"] = "1"

env = JumpKingStraightRayEnv()

model = PPO.load(
    "checkpointStraightRay/straightRay_660000_steps",
    env=env
)

obs, _ = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()