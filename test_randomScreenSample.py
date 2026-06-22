import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from jumpKingRandomScreenSample import JumpKingRandomScreenSample

env = JumpKingRandomScreenSample()

model = PPO.load(
    "checkpointRandomSample/randomScreen__95000_steps", 
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

    print(
        "level=",
        env.start_level,
        "reward=",
        reward,
        "terminated=",
        terminated,
        "truncated=",
        truncated
    )

    if terminated or truncated:

       # print("=== NUEVO EPISODIO ===")

        obs, info = env.reset()