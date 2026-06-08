#!/usr/env/bin python
#   
# Game Screen
# 
import os
import math
from turtle import done
from unittest import result

from JumpPredictionResult import JumpPredictionResult

from matplotlib.pyplot import grid
from sympy import false, true
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

		#self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		#self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)


		preview_height = 160

		screen_width = int(os.environ.get("screen_width"))
		screen_height = int(os.environ.get("screen_height"))
		window_scale = int(os.environ.get("window_scale"))

		# ventana física más alta
		self.screen = pygame.display.set_mode(
			(
				screen_width * window_scale,
				(screen_height + preview_height) * window_scale
			),
			pygame.HWSURFACE | pygame.DOUBLEBUF
		)

		# superficie interna del juego NO cambia
		self.game_screen = pygame.Surface(
			(
				screen_width,
				screen_height
			),
			pygame.HWSURFACE | pygame.DOUBLEBUF
		)

		self.preview_height = preview_height

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
		self.debug_rays = []
		self.debug_ground_rays = []
		self.cached_jump_state = [
			0.0,
			0.0,
			0.0,
			0.0
		]


		pygame.display.set_caption('Pot la IA esdevenir el Jump King ? ')

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

		# NO limitar FPS durante entrenamiento
		# self.clock.tick(self.fps)

		self._check_events()

		juegoYO = false

		if juegoYO:

			if not os.environ["pause"]:
				self._update_gamestuff()

		else:

			if not os.environ["pause"]:
				self._update_gamestuff(action=action)

		# =====================================
		# DIBUJAR SI JUEGA EL HUMANO
		# O SI EL RENDER ESTÁ ACTIVADO
		# =====================================

		if juegoYO or os.environ.get("render", "0") == "1":

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()

			pygame.display.update()

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
			#self.clock.tick(self.fps)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

		self.debug_rays = []

		self.debug_ground_rays = []

		

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

			
		self.get_state()
		self.draw_debug_rays()
		self.menus.blitme()

		# ==========================================
		# PREVIEW NIVEL SUPERIOR
		# ==========================================

		self.screen.fill((0, 0, 0))

		next_level_id = self.levels.current_level + 1

		if next_level_id in self.levels.levels:

			scale = int(os.environ.get("window_scale"))

			preview_height_px = 160 * scale

			next_level = self.levels.levels[next_level_id]

			VISIBLE_START = 200

			for p in next_level.platforms:

				rect = p.rect

				# ignorar todo lo que no esté en los
				# últimos 160 px del nivel siguiente
				if rect.bottom < VISIBLE_START:
					continue

				preview_rect = pygame.Rect(

					int(rect.x * scale),

					int((rect.y - VISIBLE_START) * scale),

					int(rect.width * scale),

					int(rect.height * scale)

				)

				pygame.draw.rect(
					self.screen,
					(0, 255, 255),
					preview_rect
				)


			# ==========================================
			# CONTINUACIÓN REAL DE LOS RAYOS
			# ==========================================

			for points, collision_type in self.debug_rays:

				visible_segments = []
				current_segment = []

				for px, py in points:

					if -160 <= py <= 0:

						current_segment.append(
							(
								int(px * scale),
								int((py + 160) * scale)
							)
						)

					else:

						if len(current_segment) >= 2:
							visible_segments.append(
								current_segment
							)

						current_segment = []

				if len(current_segment) >= 2:

					visible_segments.append(
						current_segment
					)

				# ==========================
				# DIBUJAR SEGMENTOS VISIBLES
				# ==========================

				for segment in visible_segments:

					pygame.draw.lines(
						self.screen,
						(0, 255, 0),
						False,
						segment,
						2
					)

				# ==========================
				# CRUZ FINAL
				# ==========================

				real_end_x, real_end_y = points[-1]

				if -160 <= real_end_y <= 0:

					end_x = int(real_end_x * scale)
					end_y = int((real_end_y + 160) * scale)

					size = 5

					if collision_type == "wall":
						color = (0, 100, 255)

					elif collision_type == "floor":
						color = (150, 150, 150)

					elif collision_type == "ceiling":
						color = (255, 0, 0)

					else:
						color = (255, 255, 255)

					pygame.draw.line(
						self.screen,
						color,
						(end_x - size, end_y - size),
						(end_x + size, end_y + size),
						2
					)

					pygame.draw.line(
						self.screen,
						color,
						(end_x - size, end_y + size),
						(end_x + size, end_y - size),
						2
					)

				# ==========================
				# PUNTO ORIGEN
				# ==========================

				if len(visible_segments) > 0:

					start_x, start_y = visible_segments[0][0]

					pygame.draw.circle(
						self.screen,
						(0, 150, 255),
						(start_x, start_y),
						3
					)
		scale = int(os.environ.get("window_scale"))

		self.screen.blit(
			pygame.transform.scale(
				self.game_screen,
				(
					int(os.environ.get("screen_width")) * scale,
					int(os.environ.get("screen_height")) * scale
				)
			),
			(
					self.game_screen_x,
					self.preview_height * scale
			)
		)

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
	
	

	
	

	def get_ray_rect(self, x, local_py):

		return pygame.Rect(
			int(x - 1),
			int(local_py - self.king.rect.height // 1.2 ),
			2,
			self.king.rect.height * 1.2
		)
	

	def classify_collision(
		self,
		ghost_rect,
		rect,
		prev_local_py
	):

		if (
			ghost_rect.bottom >= rect.top
			and prev_local_py < rect.top
		):

			return {
				"kind": "floor",
				"surface_y": rect.top
			}

		elif (
			ghost_rect.top <= rect.bottom
			and prev_local_py > rect.bottom
		):

			return {
				"kind": "ceiling",
				"surface_y": rect.bottom
			}

		return {
			"kind": "wall",
			"surface_y": None
		}
	
	def handle_wall_bounce(
		self,
		x,
		y,
		vx,
		rect
	):

		vx = -vx * 0.55

		if abs(x - rect.left) < abs(x - rect.right):

			x = rect.left - 2

			side = "left"

		else:

			x = rect.right + 2

			side = "right"

		return {
			"x": x,
			"y": y,
			"vx": vx,
			"side": side
		}
	

	def find_collision(
		self,
		x,
		y,
		prev_y,
		current_level,
		level_height
	):

		levels_to_check = [
			current_level,
			current_level + 1
		]

		for lvl in levels_to_check:

			if not (0 <= lvl < len(self.levels.levels)):
				continue

			local_py = y + (
				(lvl - current_level)
				* level_height
			)

			prev_local_py = prev_y + (
				(lvl - current_level)
				* level_height
			)

			for p in self.levels.levels[lvl].platforms:

				rect = p.rect

				ghost_rect = self.get_ray_rect(
					x,
					local_py
				)

				if not ghost_rect.colliderect(rect):
					continue

				collision_info = self.classify_collision(
					ghost_rect,
					rect,
					prev_local_py
				)

				return {
					"collision_kind": collision_info["kind"],
					"surface_y": collision_info["surface_y"],
					"rect": rect,
					"local_py": local_py,
					"level": lvl,
					"hit_x": x,
					"hit_y": local_py
				}

		return None
		

	def evaluate_jump(
		self,
		x,
		y,
		vx,
		vy,
		max_steps=120,
		gravity=0.27
	):

		current_level = self.king.levels.current_level
		level_height = 360

		points = [(x, y)]
		result = JumpPredictionResult()

		origin_y = y
		has_bounced = False

		for _ in range(max_steps):

			prev_y = y

			x += vx
			y += vy

			vy += gravity

			points.append((x, y))
			result.points = points

			if (
				x < 0
				or x >= 480
				or y < -360
				or y > 720
			):

				if os.environ.get("render", "0") == "1":

					self.debug_rays.append(
						(points, "none")
					)

				return  result

			collision = self.find_collision(
				x,
				y,
				prev_y,
				current_level,
				level_height
			)

			if collision is None:
				continue

			collision_kind = collision["collision_kind"]
			rect = collision["rect"]
			local_py = collision["local_py"]

			if collision_kind == "floor":
				

				result.collision_type = "floor"

				surface_y = collision["surface_y"]

				relative_height = (
					origin_y - surface_y
				) / 360.0

				if relative_height > 0:

					result.valid_floor = True

					result.relative_height = min(
						relative_height,
						1.0
					)

				if os.environ.get("render", "0") == "1":

					self.debug_rays.append(
						(
							points,
							"floor"
						)
					)
				result.collision_type = "floor"

				return result

			elif collision_kind == "ceiling":
				result.collision_type = "ceiling"
				result.hit_ceiling = True

				vy = 0.0
				vx *= 0.5
				y += 2

				continue

			elif collision_kind == "wall":

				if not has_bounced:
					result.wall_bounces += 1

					if os.environ.get("render", "0") == "1":

						self.debug_rays.append(
							(points.copy(), "wall")
						)

					bounce = self.handle_wall_bounce(
						x,
						y,
						vx,
						rect
					)

					x = bounce["x"]
					y = bounce["y"]
					vx = bounce["vx"]

					has_bounced = True

					points.append((x, y))
					result.points = points

					continue

				else:

					result.collision_type = "wall"

					if os.environ.get("render", "0") == "1":

						self.debug_rays.append(
							(points.copy(), "wall")
						)

					return result

		if os.environ.get("render", "0") == "1":

			self.debug_rays.append(
				(points, "none")
			)

		return result

	def cast_ground_ray(
		self,
		x,
		y,
		sensor_name,
		max_distance=150
	):

		current_level = self.king.levels.current_level

		points = []

		for d in range(max_distance):

			test_y = y + d

			points.append((x, test_y))

			collision = self.find_collision(
				x,
				test_y,
				test_y - 1,
				current_level,
				360
			)

			if collision is None:
				continue

			if collision["collision_kind"] == "floor":

				if os.environ.get("render", "0") == "1":

					self.debug_ground_rays.append(
						(points.copy(), "floor")
					)

				return 1.0 - (d / max_distance)

			if os.environ.get("render", "0") == "1":

				self.debug_ground_rays.append(
					(points.copy(), sensor_name)
				)

		return 0.0
	

	def get_ground_sensors(self):

		shoulder_y = (
			self.king.rect.top
			+ self.king.rect.height * 0.3
		)

		left_x = self.king.rect.left 

		center_x = self.king.rect.centerx

		right_x = self.king.rect.right 

		left_ground = self.cast_ground_ray(
			left_x,
			shoulder_y,
			"left"
		)

		center_ground = self.cast_ground_ray(
			center_x,
			shoulder_y,
			"center"
		)

		right_ground = self.cast_ground_ray(
			right_x,
			shoulder_y,
			"right"
		)

		return (
			left_ground,
			center_ground,
			right_ground
		)
	

	def get_ray_jump_vector(self, jump_count, direction):

		speed = (
			1.5 +
			((jump_count / 5) ** 1.13)
		)

		if direction == "up":

			angle = 0

		else:

			angle = (
				self.king.jumpAngles[direction]
				*
				(1 - jump_count / 45.5)
			)

			speed += 0.9

		# exactamente igual que add_vectors(0,0,...)
		x = math.sin(angle) * speed
		y = math.cos(angle) * speed

		angle = math.pi/2 - math.atan2(y, x)
		length = math.hypot(x, y)

		# convertir a velocidades cartesianas
		vx = math.sin(angle) * length
		vy = -math.cos(angle) * length

		return vx, vy
			
	
	def get_state(self):

		self.debug_ground_rays.clear()

		self.get_ground_sensors()

		x = self.king.rect.centerx
		y = self.king.rect.bottom - 4

		state = []

		jump_counts = [
		10,20,25,30
	]
		

		if not hasattr(self, "_printed_rays"):

			for jump_count in jump_counts:

				vx, vy = self.get_ray_jump_vector(
					jump_count,
					"right"
				)

				print(
					"RIGHT",
					jump_count,
					round(vx, 2),
					round(vy, 2)
				)

			self._printed_rays = True

		for jump_count in jump_counts:

			# izquierda
			vx, vy = self.get_ray_jump_vector(
				jump_count,
				"left"
			)

			result = self.evaluate_jump(
				x,
				y,
				vx,
				vy
			)

			state.append(
				1.0 if result.valid_floor else 0.0
			)

			state.append(
				result.relative_height
			)
	

			# derecha
			vx, vy = self.get_ray_jump_vector(
				jump_count,
				"right"
			)

			result = self.evaluate_jump(
				x,
				y,
				vx,
				vy
			)

			state.append(
				1.0 if result.valid_floor else 0.0
			)

			state.append(
				result.relative_height
			)

		# altura global normalizada
		level = self.king.levels.current_level
		height = self.get_global_height(level, y)

		state.append(np.tanh(height / 2000.0))

		# puede actuar
		state.append(
			1.0 if self.move_available() else 0.0
		)

		# cayendo
		state.append(
			1.0 if self.king.isFalling else 0.0
		)

		# velocidad vertical aproximada
		if not hasattr(self, "prev_y"):
			self.prev_y = y

		velocity = y - self.prev_y

		state.append(
			np.tanh(velocity / 20.0)
		)

		self.prev_y = y

		return np.array(state, dtype=np.float32)
	
	
	def draw_debug_rays(self):

		for points, collision_type in self.debug_rays:


			if len(points) < 2:
				continue

			int_points = [
				(int(px), int(py))
				for px, py in points
			]

			# trayectoria verde
			pygame.draw.lines(
				self.game_screen,
				(0, 255, 0),
				False,
				int_points,
				2
			)

			end_x, end_y = int_points[-1]

			size = 5

			# color según impacto
			if collision_type == "wall":
				color = (0, 100, 255)      # azul

			elif collision_type == "floor":
				color = (150, 150, 150)   # gris

			elif collision_type == "ceiling":
				color = (255, 0, 0)       # rojo

			else:
				color = (255, 255, 255)

			# dibujar X
			pygame.draw.line(
				self.game_screen,
				color,
				(end_x - size, end_y - size),
				(end_x + size, end_y + size),
				2
			)

			pygame.draw.line(
				self.game_screen,
				color,
				(end_x - size, end_y + size),
				(end_x + size, end_y - size),
				2
			)

			# punto origen
			start_x, start_y = int_points[0]

			pygame.draw.circle(
				self.game_screen,
				(0, 150, 255),
				(start_x, start_y),
				3
			)

		for points, sensor_name in self.debug_ground_rays:

			if len(points) < 2:
				continue

			if sensor_name == "left":

				color = (255, 0, 0)

			elif sensor_name == "center":

				color = (0, 255, 0)

			else:

				color = (0, 150, 255)

			pygame.draw.lines(
				self.game_screen,
				color,
				False,
				[(int(px), int(py)) for px, py in points],
				2
			)	