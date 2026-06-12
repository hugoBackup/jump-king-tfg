from JumpKingMulti import JumpKingMulti
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


env = JumpKingMulti()

model = PPO.load(
	"JumpKingMatrix2.0",
	env=env
)

checkpoint_callback = CheckpointCallback(
	save_freq=10_000,
	save_path="./checkpoints",
	name_prefix="matrix2.5"
)

model.learn(
	total_timesteps=300_000,
	callback=checkpoint_callback,
	reset_num_timesteps=False
)

model.save(
	"jumpkingMatrix2.5"
)