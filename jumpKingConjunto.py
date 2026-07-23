import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from JumpKing import JKGame


class jumpKingConjunto(gym.Env):

	def __init__(
		self,
		training_levels=None
	):

		super().__init__()

		self.game = JKGame(
			max_step=1000
		)

		self.action_space = spaces.Discrete(10)

		sample_state = np.zeros(
			38,
			dtype=np.float32
		)

		self.observation_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=sample_state.shape,
			dtype=np.float32
		)

		self.current_step = 0

		self.cached_jump_state = [0.0] * 30

		self.cached_ground_state = [
			0.0,
			0.0,
			0.0
		]

		self.spawn_points = {
			0: (200, 250),
			1: (350, 250),
			5: (250, 250),
			12: (400, 200),
			13: (220, 200),
			14: (150, 250),
			18: (380, 250),
			19: (260, 220),
			20: (250, 250),
			42: (120, 250),
		}

		if training_levels is None:

			self.training_levels = list(
				self.spawn_points.keys()
			)

		else:

			invalid_levels = [
				lvl
				for lvl in training_levels
				if lvl not in self.spawn_points
			]

			if len(invalid_levels) > 0:

				raise ValueError(
					f"Pantallas no definidas en spawn_points: {invalid_levels}"
				)

			self.training_levels = (
				training_levels
			)

	def reset(self, seed=None, options=None):

		self.reset_lock = 20   # si el juego va a 60 FPS
		

		super().reset(seed=seed)
		print("(///////////////////RESET/////////////////////")
		self.current_step = 0

		self.episode_reward = 0.0

		self.game.reset()

		self.start_level = random.choice(
			self.training_levels
		)

		x, y = self.spawn_points[
			self.start_level
		]

		self.game.levels.current_level = (
			self.start_level
		)

		self.game.king.rect_x = x
		self.game.king.rect_y = y

		self.best_height = self.game.get_global_height(
			self.start_level,
			y
		)

		if self.start_level == 42:

			goal_x = 300
			goal_y = 100

			self.best_distance = np.sqrt(
				(x - goal_x) ** 2 +
				(y - goal_y) ** 2
			)

		if hasattr(self, "prev_y"):
			del self.prev_y

		return self.get_state(), {}

	def step(self, action):

		if self.reset_lock > 0:
			self.reset_lock -= 1

			self.game.step(None)

			return (
				self.get_state(),
				0.0,
				False,
				False,
				{}
			)

		self.game.last_action = int(action)

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

		self.current_step += 1

		new_level = (
			self.game.king.levels.current_level
		)

		new_height = self.game.get_global_height(
			new_level,
			self.game.king.y
		)

		height_gain = new_height - old_height

		reward = -0.1

		terminated = False
		truncated = False

		# ======================
		# PROGRESO VERTICAL
		# ======================

		if new_height > self.best_height:

			reward += (
				new_height - self.best_height
			) / 10.0

			self.best_height = new_height

		# ======================
		# NIVEL 42
		# ======================

		if self.start_level == 42:

			x = self.game.king.rect.centerx
			y = self.game.king.y

			goal_x = 300
			goal_y = 100

			distance = np.sqrt(
				(x - goal_x) ** 2 +
				(y - goal_y) ** 2
			)

			if distance < self.best_distance:

				reward += (
					self.best_distance - distance
				) / 10.0

				self.best_distance = distance

			if distance < 1:

				reward = 100.0
				terminated = True

			if new_level < 42:

				reward = -100.0
				terminated = True

		# ======================
		# RESTO DE NIVELES
		# ======================

		else:

			if new_level > self.start_level:

				reward = 100.0
				terminated = True

			elif new_level < self.start_level:

				reward = -100.0
				terminated = True

		# ======================
		# TIMEOUT
		# ======================

		if (
			not terminated
			and self.current_step >= 32
		):

			reward = -100.0
			truncated = True

		action_names = {
			0: "J5L",
			1: "J5R",
			2: "J10L",
			3: "J10R",
			4: "J20L",
			5: "J20R",
			6: "J25L",
			7: "J25R",
			8: "J30L",
			9: "J30R"
		}

		self.episode_reward += reward
		
		print(f"L{new_level} | "f"{action_names[int(action)]} | "f"height={new_height:.1f} | "f"gain={height_gain:.1f} | "f"best={self.best_height:.1f} | "f"reward={reward:.3f} | "f"total={self.episode_reward:.3f}")

		return (
			self.get_state(),
			reward,
			terminated,
			truncated,
			{}
		)

	def get_state(self):

		x = self.game.king.rect.centerx

		y = (
			self.game.king.rect.bottom - 7
		)

		state = []

		if self.game.move_available():

			jump_counts = [
				5,
				10,
				20,
				25,
				30
			]

			jump_state = []

			for jump_count in jump_counts:

				for direction in [
					"left",
					"right"
				]:

					vx, vy = (
						self.game.get_ray_jump_vector(
							jump_count,
							direction
						)
					)

					result = (
						self.game.evaluate_jump(
							x,
							y,
							vx,
							vy
						)
					)

					jump_state.append(
						1.0
						if result.valid_floor
						else 0.0
					)

					jump_state.append(
						result.relative_height
					)

					jump_state.append(
						getattr(
							result,
							"relative_x",
							0.0
						)
					)

			self.cached_jump_state = (
				jump_state
			)

		state.extend(
			self.cached_jump_state
		)

		if self.game.move_available():

			self.cached_ground_state = list(
				self.game.get_ground_sensors()
			)

		state.extend(
			self.cached_ground_state
		)

		level = (
			self.game.king.levels.current_level
		)

		height = (
			self.game.get_global_height(
				level,
				y
			)
		)

		state.append(
			np.tanh(
				height / 2000.0
			)
		)

		state.append(
			1.0
			if self.game.move_available()
			else 0.0
		)

		state.append(
			1.0
			if self.game.king.isFalling
			else 0.0
		)

		if not hasattr(self, "prev_y"):
			self.prev_y = y

		velocity = y - self.prev_y

		state.append(
			np.tanh(
				velocity / 20.0
			)
		)

		state.append(
			np.tanh(
				(x - 240) / 240.0
			)
		)

		self.prev_y = y

		return np.array(
			state,
			dtype=np.float32
		)

	def execute_action(self, action):

		if action == 0:  # J5L

			for _ in range(5):
				self.game.step(3)

			self.game.step(1)

		elif action == 1:  # J5R

			for _ in range(5):
				self.game.step(2)

			self.game.step(0)

		elif action == 2:  # J10L

			for _ in range(10):
				self.game.step(3)

			self.game.step(1)

		elif action == 3:  # J10R

			for _ in range(10):
				self.game.step(2)

			self.game.step(0)

		elif action == 4:  # J20L

			for _ in range(20):
				self.game.step(3)

			self.game.step(1)

		elif action == 5:  # J20R

			for _ in range(20):
				self.game.step(2)

			self.game.step(0)

		elif action == 6:  # J25L

			for _ in range(25):
				self.game.step(3)

			self.game.step(1)

		elif action == 7:  # J25R

			for _ in range(25):
				self.game.step(2)

			self.game.step(0)

		elif action == 8:  # J30L

			for _ in range(30):
				self.game.step(3)

			self.game.step(1)

		elif action == 9:  # J30R

			for _ in range(30):
				self.game.step(2)

			self.game.step(0)

	def set_training_levels(self, levels):

		invalid_levels = [
			lvl for lvl in levels
			if lvl not in self.spawn_points
		]

		if invalid_levels:
			raise ValueError(
				f"Pantallas no definidas en spawn_points: {invalid_levels}"
			)

		self.training_levels = levels