import os
import math

import numpy as np

from JumpKing import JumpPredictionResult
from JumpKing import JKGame

class jumpKingAgentCurvedRay:

	def __init__(self, game):

		self.observation_size = 38

		self.game = game

	def reset(self):

		self.prev_y = None
		self.cached_jump_state = [-1.0] * 30
		self.cached_ground_state = [0.0] * 3	
		
	def get_state(self):

		self.game.debug_ground_rays.clear()

		x = self.game.king.rect.centerx
		y = self.game.king.rect.bottom - 4

		state = []

		if self.game.move_available():

			jump_state = []

			jump_counts = [
				5,
				10,
				20,
				25,
				30
			]

			for jump_count in jump_counts:

				for direction in [
					"left",
					"right"
				]:

					vx, vy = self.get_ray_jump_vector(
						jump_count,
						direction
					)

					result = self.evaluate_jump(
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

					jump_state.append(
						result.relative_x
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
				self.get_ground_sensors()
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

		if self.prev_y is None:

			self.prev_y = y

		velocity = y - self.prev_y

		state.append(
			np.tanh(
				velocity / 20.0
			)
		)

		# =====================
		# POSICIÓN X
		# =====================

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
		
	def get_ray_jump_vector(self, jump_count, direction):

		speed = (
			1.5 +
			((jump_count / 5) ** 1.13)
		)

		if direction == "up":

			angle = 0

		else:

			angle = (
				self.game.king.jumpAngles[direction]
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
	
	def evaluate_jump(
		self,
		x,
		y,
		vx,
		vy,
		max_steps=120,
		gravity=0.27
	):
		
		if (
			not self.game.move_available()
			and os.environ.get("render", "0") == "1"
		):
			return JumpPredictionResult()

		current_level = self.game.king.levels.current_level
		level_height = 360

		

		points = [(x, y)]
		result = JumpPredictionResult()

		origin_y = y
		origin_x = x
		has_bounced = False
		has_hit_ceiling = False

		for step in range(max_steps):

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

					self.game.debug_rays.append(
						(points, "none")
					)

				return  result

			collision = self.game.find_collision(
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

			if (
				collision_kind == "floor"
				and vy < 0
				and step < 5
			):
				collision_kind = "wall"

			if collision_kind == "floor":

				result.collision_type = "floor"

				surface_y = collision["surface_y"]

				level_diff = collision["level"] - current_level
				surface_y_local = surface_y - (level_diff * 360)

				relative_height = (
					origin_y - surface_y_local
				) / 360.0

				if relative_height > 0:

					result.valid_floor = True

					result.landing_x = x
					result.landing_y = surface_y - 10
					result.landing_level = collision["level"]

					result.relative_height = min(
						relative_height,
						1.0
					)

					result.relative_x = np.clip(
						(x - origin_x) / 480.0,
						-1.0,
						1.0
					)

				if os.environ.get("render", "0") == "1":

					self.game.debug_rays.append(
						(
							points,
							"floor"
						)
					)

				return result

			elif collision_kind == "ceiling":

				if not has_hit_ceiling:

					if os.environ.get("render", "0") == "1":

						self.game.debug_rays.append(
							(points.copy(), "ceiling")
						)

					result.hit_ceiling = True

					vy = 0.0
					vx *= 0.2 #antes estaba en 0.5 pero creo que esto es mas preciso , ahora mismo hay rayos que presentan trayectorias malas
					y += 2

					has_hit_ceiling = True

				continue

			elif collision_kind == "wall":

				if step < 2:
					continue

				if not has_bounced:
					result.wall_bounces += 1

					if os.environ.get("render", "0") == "1":

						self.game.debug_rays.append(
							(points.copy(), "wall")
						)


					
					bounce = self.game.handle_wall_bounce(
						x,
						y,
						vx,
						vy,
						rect
					)

					x = bounce["x"]
					y = bounce["y"]
					vx = bounce["vx"]
					vy = bounce["vy"]

					has_bounced = True

					points.append((x, y))
					result.points = points

					continue

				else:

					result.collision_type = "wall"

					if os.environ.get("render", "0") == "1":

						self.game.debug_rays.append(
							(points.copy(), "wall")
						)

					return result

		if os.environ.get("render", "0") == "1":

			self.game.debug_rays.append(
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

		current_level = self.game.king.levels.current_level

		points = []

		for d in range(max_distance):

			test_y = y + d

			points.append((x, test_y))

			collision = self.game.find_collision(
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

					self.game.debug_ground_rays.append(
						(points.copy(), "floor")
					)

				return 1.0 - (d / max_distance)

			if os.environ.get("render", "0") == "1":

				self.game.debug_ground_rays.append(
					(points.copy(), sensor_name)
				)

		return 0.0
	
	def get_ground_sensors(self):

		shoulder_y = (
			self.game.king.rect.top
			+ self.game.king.rect.height * 0.3
		)

		left_x = self.game.king.rect.left
		center_x = self.game.king.rect.centerx
		right_x = self.game.king.rect.right

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
			1.0 if left_ground > 0.5 else 0.0,
			1.0 if center_ground > 0.5 else 0.0,
			1.0 if right_ground > 0.5 else 0.0
		)
				