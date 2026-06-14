import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from JumpKingMulti import JumpKingMulti
from JumpKingSinRL import JumpKingSinRL

env = JumpKingSinRL()

model = PPO.load(
"jumpkingSinRLMlp",
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

