
import os
from stable_baselines3 import PPO

from envs.JumpkingcurvedRayEnv import JumpKingCurvedRayEnv
from envs.jumpKingStraightRayEnv import JumpKingStraightRayEnv

TRAIN = True
os.environ["render"] = "1"
MODEL_NAME = "jumpKingStraightRay"

env = JumpKingStraightRayEnv()

if TRAIN:

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64
    )

    model.learn(
        total_timesteps=10_000
    )

    model.save(MODEL_NAME)

else:

    model = PPO.load(
        MODEL_NAME,
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