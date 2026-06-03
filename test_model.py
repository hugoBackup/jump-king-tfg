import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from jumpKingLongo import JumpKingLongo

env = JumpKingLongo()

model = PPO.load("ppo_jumpking_discreto")

obs, _ = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, _ = env.step(
        action
    )

    if terminated or truncated:

        print("RESET")

        obs, _ = env.reset()