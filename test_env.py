from JumpKingMulti import JumpKingMulti
from JumpKingSinRL import JumpKingSinRL
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


env = JumpKingSinRL()

print("////////////////MlpPolicy////////////////////")

model = PPO(
	"MlpPolicy",
	env,
	verbose=1,
	learning_rate=3e-4,
	n_steps=512,
	batch_size=64
)

checkpoint_callback = CheckpointCallback(
	save_freq=10_000,
	save_path="./checkpointSinRL",
	name_prefix="sinRL"
)

model.learn(
	total_timesteps=500_000,
	callback=checkpoint_callback
)

model.save(
	"jumpkingSinRLMlp"
)