import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from JumpKingMulti import JumpKingMulti

env = JumpKingMulti()

model = PPO.load(
"ppo_jumpking_multi",
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

