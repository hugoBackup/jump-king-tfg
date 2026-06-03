import gymnasium as gym
from gymnasium import spaces
import numpy as np

from JumpKing import JKGame


class JumpKingEnvDiscreto(gym.Env):

	def __init__(self, render_mode=False):

		super().__init__()

		self.render_mode = render_mode

		self.game = JKGame(max_step=1000)

		# =====================================
		# ACCIONES DISCRETAS
		# =====================================

		self.action_space = spaces.Discrete(15)

		self.game.reset()

		sample_state = self.game.get_state()

		self.observation_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=sample_state.shape,
			dtype=np.float32
		)

		self.current_step = 0


	# =====================================
	# RESET
	# =====================================

	def reset(self, seed=None, options=None):

		state = self.game.reset()

		self.current_step = 0

		return np.array(state, dtype=np.float32), {}


	# =====================================
	# STEP
	# =====================================

	def step(self, action):

		old_level = self.game.king.levels.current_level
		old_y = self.game.king.y

		old_height = self.game.get_global_height(
			old_level,
			old_y
		)

		self.execute_action(action)

		max_sim_steps = 200
		sim_steps = 0

		while (
			not self.game.move_available()
			and sim_steps < max_sim_steps
		):

			self.game.step(None)

			sim_steps += 1

		final_level = self.game.king.levels.current_level
		final_y = self.game.king.y

		new_height = self.game.get_global_height(
			final_level,
			final_y
		)

		# =====================================
		# RECOMPENSA
		# =====================================

		delta = new_height - old_height

		reward = delta / 50.0

		if final_level > old_level:

			reward += 20.0

		elif final_level < old_level:

			reward -= 20.0

		state = np.array(
			self.game.get_state(),
			dtype=np.float32
		)

		self.current_step += 1

		truncated = self.current_step >= 1000
		terminated = False

		return (
			state,
			reward,
			terminated,
			truncated,
			{}
		)


	# =====================================
	# ACCIONES
	# =====================================

	def execute_action(self, action):

		action = int(action)

		# ---------------------------------
		# caminar izquierda
		# ---------------------------------

		if action == 0:

			for _ in range(8):

				self.game.step(1)

			return

		# ---------------------------------
		# caminar derecha
		# ---------------------------------

		if action == 1:

			for _ in range(8):

				self.game.step(0)

			return

		# ---------------------------------
		# esperar
		# ---------------------------------

		if action == 14:

			for _ in range(8):

				self.game.step(None)

			return

		# ---------------------------------
		# saltos
		# ---------------------------------

		jump_table = {

			2: ("left", 5),
			3: ("right", 5),

			4: ("left", 10),
			5: ("right", 10),

			6: ("left", 15),
			7: ("right", 15),

			8: ("left", 20),
			9: ("right", 20),

			10: ("left", 25),
			11: ("right", 25),

			12: ("left", 30),
			13: ("right", 30),
		}

		direction, charge = jump_table[action]

		hold = 3 if direction == "left" else 2
		release = 1 if direction == "left" else 0

		for _ in range(charge):

			self.game.step(hold)

		self.game.step(release)