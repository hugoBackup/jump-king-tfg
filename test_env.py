from stable_baselines3 import PPO
from JumpKingEnvContinous import JumpKingEnvContinuous

env = JumpKingEnvContinuous()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=100_000)

model.save("ppo_jumpking")


# TEST
obs, _ = env.reset()

for step in range(300):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)

    print(f"Step {step} | Reward: {reward:.2f}")

    # 🔥 reset solo si realmente cortas episodio
    if truncated:
        print("Reset por límite de pasos")
        obs, _ = env.reset()