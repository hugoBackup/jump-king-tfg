import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from JumpKing import JKGame
from Logs.training_logger import TrainingLogger


class JumpKingPunishEnv(gym.Env):

	def __init__(self, agent_class):

		

		super().__init__()

		

		self.game = JKGame(
			max_step=100
		)
		self.agent = agent_class(self.game)
		self.action_space = spaces.Discrete(10)
		self.game.agent = self.agent

		self.game.reset()

		sample_state = np.zeros(
			self.agent.observation_size,
			dtype=np.float32
		)

		self.observation_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=sample_state.shape,
			dtype=np.float32
		)

		self.logger = TrainingLogger(
			self.agent.__class__.__name__
		)

		self.current_step = 0

		# exploración de plataforma
		self.visited_positions = set()

		# recompensa Fibonacci
		self.consecutive_good_jumps = 0


	def reset(self, seed=None, options=None):


		if self.current_step > 0:

			self.logger.log_episode(
				reward=self.episode_reward,
				max_height=self.best_height,
				max_level=self.best_level,
				actions=self.action_counter,
				actions_to_max_height=self.max_height_actions,
				unique_positions=len(self.visited_positions)
			)
		#print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")



		super().reset(seed=seed)

		

		

		self.action_counter = 0

		self.episode_reward = 0.0

		self.current_step = 0

		self.total_height_gain = 0.0

		self.visited_positions.clear()

		self.consecutive_good_jumps = 0
		self.same_height_jump_counter = 0
		self.same_height_counter = 0
		self.last_height = None

		self.game.reset()

		self.best_height = self.game.get_global_height(
			self.game.king.levels.current_level,
			self.game.king.y
		)

		self.max_height_actions = 0
		self.best_level = self.game.king.levels.current_level
		if hasattr(self.agent, "reset"):
			self.agent.reset()

		return self.get_state(), {}

	def step(self, action):

		#print("ACTION RECIBIDA:", action)

		old_level = self.game.king.levels.current_level
		old_y = self.game.king.y

		old_height = self.game.get_global_height(
			old_level,
		
			old_y
		)

		
		self.game.last_action = int(action)
		self.execute_action(action)		
		self.action_counter += 1

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

		reward = -0.5

		if self.same_height_jump_counter >= 3:

			reward -= 1.0

		# ==================================
		# EXPLORACIÓN
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
		# PROGRESO VERTICAL
		# ==================================

		height_gain = (
			new_height - old_height
		)
		self.total_height_gain += max(0, height_gain)

		 #antes era si accion es mayor que 2 pero ahora y ano puede caminar asi que asi se queda

		if abs(height_gain) < 1:

			self.same_height_jump_counter += 1

		else:

			self.same_height_jump_counter = 0

		

		# ==================================
		# PENALIZAR CAÍDAS
		# ==================================

		if height_gain < 0:

			reward -= 1.0

			reward += height_gain / 100.0

		# ==================================
		# SALTOS BUENOS
		# ==================================

		if (
			height_gain > 5
		):

			self.consecutive_good_jumps += 1

			base_reward = (
				2.0 *
				self.fibonacci(
					self.consecutive_good_jumps
				)
			)

			if new_height >= self.best_height:

				self.best_height = new_height
				
				self.max_height_actions = self.action_counter

				self.best_level = new_level

				reward += base_reward * 2.0

		else:

			self.consecutive_good_jumps = 0

		# ==================================
		# FIN EPISODIO
		# ==================================

		self.current_step += 1

		terminated = False

		truncated = (
			self.current_step >= 100
		)

		action_names = {
			0:"J5L",
			1:"J5R",
			2:"J10L",
			3:"J10R",
			4:"J20L",
			5:"J20R",
			6:"J25L",
			7:"J25R",
			8:"J30L",
			9:"J30R"
		}

		action = int(action)

		self.episode_reward += reward

		

		#print(f"{action_names[action]} | "f"height={new_height:.1f} | "f"gain={height_gain:.1f} | "f"best={self.best_height:.1f} | "f"reward={reward:.3f} | "f"total={self.episode_reward:.3f}")

		

		
		
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

		return self.agent.get_state()

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

	def close(self):

		self.logger.close()

		super().close()		