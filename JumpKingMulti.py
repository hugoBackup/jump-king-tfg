import gymnasium as gym
from gymnasium import spaces
import numpy as np

from JumpKing import JKGame


class JumpKingMulti(gym.Env):

	def __init__(self):

		super().__init__()

		self.game = JKGame(
			max_step=1000
		)

		# 0 = caminar izquierda
		# 1 = caminar derecha
		# 2 = salto largo izquierda
		# 3 = salto largo derecha
		self.action_space = spaces.Discrete(10)

		self.game.reset()

		# =====================
		# ESTADO
		# =====================
		#
		# jump_left_valid
		# jump_left_height
		#
		# jump_right_valid
		# jump_right_height
		#
		# ground_left
		# ground_center
		# ground_right
		#
		# global_height
		#
		# move_available
		#
		# is_falling
		#
		# vertical_velocity
		#
		# TOTAL = 11
		#
		sample_state = np.zeros(
			23,
			dtype=np.float32
		)

		self.observation_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=sample_state.shape,
			dtype=np.float32
		)

		self.current_step = 0

		# exploración de plataforma
		self.visited_positions = set()

		# recompensa Fibonacci
		self.consecutive_good_jumps = 0

		self.cached_jump_state = [
			0.0
		] * 16

		self.cached_ground_state = [
			0.0,
			0.0,
			0.0
		]



		

	def reset(self, seed=None, options=None):

		super().reset(seed=seed)

		self.current_step = 0

		self.visited_positions.clear()

		self.consecutive_good_jumps = 0

		self.game.reset()

		if hasattr(self, "prev_y"):

			del self.prev_y

		return self.get_state(), {}

	def step(self, action):
		print("ACTION RECIBIDA:", action)

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

		new_level = self.game.king.levels.current_level
		new_y = self.game.king.y

		new_height = self.game.get_global_height(
			new_level,
			new_y
		)

		# ==================================
		# RECOMPENSA BASE
		# ==================================

		reward = -0.01

		# ==================================
		# EXPLORACIÓN DE PLATAFORMA
		# ==================================

		bucket_x = int(
			self.game.king.rect.centerx / 8
		)

		position_key = (
			new_level,
			bucket_x
		)

		if position_key not in self.visited_positions:

			self.visited_positions.add(
				position_key
			)

			reward += 0.05

		# ==================================
		# SALTO CORRECTO
		# ==================================

		height_gain = (
			new_height - old_height
		)

		if action >= 2 and height_gain > 20:

			self.consecutive_good_jumps += 1

			reward += self.fibonacci(
				self.consecutive_good_jumps
			)

		else:

			self.consecutive_good_jumps = 0

		# ==================================
		# FIN EPISODIO
		# ==================================

		self.current_step += 1

		terminated = False

		truncated = (
			self.current_step >= 1000
		)


		action_names = {
			0: "L",
			1: "R",
			2: "J10L",
			3: "J10R",
			4: "J20L",
			5: "J20R",
			6: "J25L",
			7: "J25R",
			8: "J30L",
			9: "J30R"
		}

		print(
			f"{action_names[action]} | "
			f"height={new_height:.1f} | "
			f"gain={height_gain:.1f} | "
			f"reward={reward:.3f}"
		)

		

		return (
			self.get_state(),
			reward,
			terminated,
			truncated,
			{}
		)


	def fibonacci(self, n):

		if n <= 2:

			return 1

		a = 1
		b = 1

		for _ in range(n - 2):

			a, b = b, a + b

		return b

	def get_state(self):

		x = self.game.king.rect.centerx

		y = self.game.king.rect.bottom - 7

		state = []
		if self.game.move_available():

			jump_counts = [10, 20, 25, 30]

			jump_state = []

			for jump_count in jump_counts:

				for direction in ["left", "right"]:

					vx, vy = self.game.get_ray_jump_vector(
						jump_count,
						direction
					)

					result = self.game.evaluate_jump(
						x,
						y,
						vx,
						vy
					)

					jump_state.append(
						1.0 if result.valid_floor else 0.0
					)

					jump_state.append(
						result.relative_height
					)

			self.cached_jump_state = jump_state

		state.extend(
			self.cached_jump_state
		)
		# =====================
		# SENSORES DE SUELO
		# =====================

		if self.game.move_available():

			self.cached_ground_state = list(
				self.game.get_ground_sensors()
			)

		state.extend(
			self.cached_ground_state
		)

		# =====================
		# ALTURA GLOBAL
		# =====================

		level = self.game.king.levels.current_level

		height = self.game.get_global_height(
			level,
			y
		)

		state.append(
			np.tanh(
				height / 2000.0
			)
		)

		# =====================
		# MOVE AVAILABLE
		# =====================

		state.append(
			1.0
			if self.game.move_available()
			else 0.0
		)

		# =====================
		# FALLING
		# =====================

		state.append(
			1.0
			if self.game.king.isFalling
			else 0.0
		)

		# =====================
		# VELOCIDAD VERTICAL
		# =====================

		if not hasattr(
			self,
			"prev_y"
		):
			self.prev_y = y

		velocity = y - self.prev_y

		state.append(
			np.tanh(
				velocity / 20.0
			)
		)

		self.prev_y = y

		return np.array(
			state,
			dtype=np.float32
		)

	def execute_action(self, action):

		if action == 0:
			self.game.step(3)

		elif action == 1:
			self.game.step(2)

		elif action == 2:

			for _ in range(10):
				self.game.step(3)

			self.game.step(1)

		elif action == 3:

			for _ in range(10):
				self.game.step(2)

			self.game.step(0)

		elif action == 4:

			for _ in range(20):
				self.game.step(3)

			self.game.step(1)

		elif action == 5:

			for _ in range(20):
				self.game.step(2)

			self.game.step(0)

		elif action == 6:

			for _ in range(25):
				self.game.step(3)

			self.game.step(1)

		elif action == 7:

			for _ in range(25):
				self.game.step(2)

			self.game.step(0)

		elif action == 8:

			for _ in range(30):
				self.game.step(3)

			self.game.step(1)

		elif action == 9:

			for _ in range(30):
				self.game.step(2)

			self.game.step(0)