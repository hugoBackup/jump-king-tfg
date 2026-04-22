#!/usr/env/bin python
#   
# Game Screen
# 
import os
from turtle import done

from matplotlib.pyplot import grid
#os.environ["SDL_VIDEODRIVER"] = "dummy" esto para quitar el video 





import pygame 
import sys

import inspect
import pickle
import numpy as np
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus

from Start import Start

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time


class NETWORK(torch.nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
		"""DQN Network example
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
		super(NETWORK, self).__init__()

		self.layer1 = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.layer2 = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.final = torch.nn.Linear(hidden_dim, output_dim)
		self.current_step = 0

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.final(x)

		return x


class DDQN(object):
	def __init__(
			self
	):
		
		self.last_positions = {}
		self.target_net = NETWORK(4, 4, 32)
		self.eval_net = NETWORK(4, 4, 32)

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

		self.memory_counter = 0
		self.memory_size = 50000
		self.memory = np.zeros((self.memory_size, 11))

		self.epsilon = 1.0
		self.epsilon_decay = 0.95
		self.alpha = 0.99

		self.batch_size = 64
		self.episode_counter = 0

		self.target_net.load_state_dict(self.eval_net.state_dict())

	def memory_store(self, s0, a0, r, s1, sign):
		transition = np.concatenate((s0, [a0, r], s1, [sign]))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def select_action(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		if np.random.uniform() > self.epsilon:
			logit = self.eval_net(state)
			action = torch.argmax(logit, 1).item()
		else:
			action = int(np.random.choice(4, 1))

		return action

	def policy(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		logit = self.eval_net(state)
		action = torch.argmax(logit, 1).item()

		return action

	def train(self, s0, a0, r, s1, sign):
		if sign == 1:
			if self.episode_counter % 2 == 0:
				self.target_net.load_state_dict(self.eval_net.state_dict())
			self.episode_counter += 1

		self.memory_store(s0, a0, r, s1, sign)
		self.epsilon = np.clip(self.epsilon * self.epsilon_decay, a_min=0.01, a_max=None)

		# select batch sample
		if self.memory_counter > self.memory_size:
			batch_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[batch_index]
		batch_s0 = torch.tensor(batch_memory[:, :4]).float()
		batch_a0 = torch.tensor(batch_memory[:, 4:5]).long()
		batch_r = torch.tensor(batch_memory[:, 5:6]).float()
		batch_s1 = torch.tensor(batch_memory[:, 6:10]).float()
		batch_sign = torch.tensor(batch_memory[:, 10:11]).long()

		q_eval = self.eval_net(batch_s0).gather(1, batch_a0)

		with torch.no_grad():
			maxAction = torch.argmax(self.eval_net(batch_s1), 1, keepdim=True)
			q_target = batch_r + (1 - batch_sign) * self.alpha * self.target_net(batch_s1).gather(1, maxAction)

		loss = self.criterion(q_eval, q_target)

		# backward
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class JKGame:
	""" Overall class to manga game aspects """

	

	def get_global_height(self, level, y):
		return level * 360 - y

	
        
	def __init__(self, max_step=float('inf')):

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)

		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):

		self.last_positions = {}
		self.visited = {}

		self.king.reset()
		self.levels.reset()

		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0

		current_level = self.king.levels.current_level
		current_y = self.king.y

		self.last_height = self.get_global_height(current_level, current_y)
		self.best_height = self.last_height

		self.prev_y = current_y

		self.visited[(current_level, int(current_y))] = 1

		state = np.array(self.get_state(), dtype=np.float32)

		return state   # ← SOLO ESTO
		
	
	def move_available(self):

		# no puede actuar en el aire
		if self.king.isFalling:
			return False

		# no puede actuar durante animación de caída
		if self.king.isSplat and self.king.splatCount <= self.king.splatDuration:
			return False

		# si el juego terminó
		if self.king.levels.ending:
			return False

		return True
	

	
	def step(self, action):

		self.clock.tick(self.fps)
		self._check_events()

		if not os.environ["pause"]:
			#if not self.move_available():
				#	action = None
			self._update_gamestuff(action=action)

		self._update_gamescreen()
		self._update_guistuff()
		self._update_audio()
		pygame.display.update()

		# estado basado en sensores (raycasting)
		state = self.get_state()

		return state
	
	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
			#state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			#print(state)
			#self.clock.tick(self.fps)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			self.clock.tick(0)
			self._check_events()
			if not os.environ["pause"]:
				self._update_gamestuff()

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()

	def _check_events(self):

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, action=None):

		self.levels.update_levels(self.king, self.babe, agentCommand=action)

	def _update_guistuff(self):

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))

	def is_jump_finished(self):
		return self.move_available()		
	
	def cast_ray(self, x, y, angle_deg, max_distance=200, step=3):

		angle = np.radians(angle_deg)

		dx = np.cos(angle)
		dy = -np.sin(angle)

		current_level = self.king.levels.current_level
		level_height = 360

		for dist in range(5, max_distance, step):

			px = x + dx * dist
			py = y + dy * dist

			# fuera de pantalla
			if px < 0 or px >= 480 or py < 0 or py >= 360:
				return dist / max_distance

			# niveles a comprobar
			levels_to_check = [current_level]

			if dy < 0:
				levels_to_check.append(current_level + 1)

			if dy > 0:
				levels_to_check.append(current_level - 1)

			for lvl in levels_to_check:
				if 0 <= lvl < len(self.levels.levels):

					# ajustar coordenadas al nivel correcto
					local_py = py - (lvl - current_level) * level_height

					for p in self.levels.levels[lvl].platforms:
						if p.rect.collidepoint(px, local_py):
							return dist / max_distance

		return 1.0
	
	def get_state(self):

		x = self.king.x
		y = self.king.y

		angles = [
			250, 270, 290,     # suelo
			0, 180,            # laterales
			315, 225,          # diagonales
			30, 60, 90, 120, 150  # arriba
		]

		state = []

		# raycasting
		for angle in angles:
			dist = self.cast_ray(x, y, angle)
			state.append(dist)

		# altura global
		level = self.king.levels.current_level
		height = self.get_global_height(level, y)
		state.append(np.tanh(height / 2000.0))

		# puede saltar
		state.append(1.0 if self.move_available() else 0.0)

		# cayendo
		state.append(1.0 if self.king.isFalling else 0.0)

		# velocidad (segura)
		if not hasattr(self, "prev_y"):
			self.prev_y = y

		velocity = y - self.prev_y
		state.append(velocity / 20.0)

		# actualizar memoria
		self.prev_y = y

		return np.array(state, dtype=np.float32)
