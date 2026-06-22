import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from jumpKingRandomScreen import JumpKingRandomScreen

env = JumpKingRandomScreen()

model = PPO.load(
    "checkpointRandomDesde0/randomScreen__452000_steps",
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

    print("reward=", reward,"terminated=", terminated, "truncated=",truncated)

    if terminated or truncated:

       # print("=== NUEVO EPISODIO ===")

        obs, info = env.reset()