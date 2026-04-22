from stable_baselines3 import PPO
from JumpKingEnvContinous import JumpKingEnvContinuous

# create env
env = JumpKingEnvContinuous()

# create model
model = PPO("MlpPolicy", env, verbose=1)

# train
model.learn(total_timesteps=10000)

# save
model.save("ppo_jumpking")

# test run
obs, _ = env.reset()

for step in range(100):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, _ = env.step(action)

    print(f"Step {step} | Reward: {reward}")

    if done:
        obs, _ = env.reset()