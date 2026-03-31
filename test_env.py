from stable_baselines3 import DQN
from jumpking_env import JumpKingEnv

env = JumpKingEnv()

#model = DQN.load("dqn_jumpking", env=env)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=1_000)

model.save("dqn_jumpking")