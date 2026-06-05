from JumpKingMulti import JumpKingMulti
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


env = JumpKingMulti()

model = PPO(
	"MlpPolicy",
	env,
	verbose=1,
	learning_rate=3e-4,
	n_steps=2048,
	batch_size=64
)

checkpoint_callback = CheckpointCallback(
	save_freq=10_000,
	save_path="./checkpoints",
	name_prefix="jumpking_multi"
)

model.learn(
	total_timesteps=100_000,
	callback=checkpoint_callback
)

model.save(
	"ppo_jumpking_multi"
)