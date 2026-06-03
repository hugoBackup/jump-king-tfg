from jumpKingLongo import JumpKingLongo
from stable_baselines3 import PPO



env = JumpKingLongo()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

model.learn(
    total_timesteps=100_000
)

model.save("ppo_conSuelo_jumpking_discreto")