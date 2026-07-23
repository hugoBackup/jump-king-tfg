import os

os.environ["render"] = "1"

from stable_baselines3 import PPO
from jumpKingConjunto import jumpKingConjunto

MODEL_PATH = (
	"checkpoint19-izquierda/conjunto19__105000_steps"
)

env = jumpKingConjunto(
	training_levels=[19]
)

model = PPO.load(
	MODEL_PATH,
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