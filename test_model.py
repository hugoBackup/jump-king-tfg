import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from JumpKingMulti import JumpKingMulti

env = JumpKingMulti()

model = PPO.load(
"checkpoints/matrix1.1_20000_steps",
env=env
)

obs, info = env.reset()

while True:


    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if terminated or truncated:

        obs, info = env.reset()

