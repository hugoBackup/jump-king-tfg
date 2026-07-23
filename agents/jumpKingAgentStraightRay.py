import math
import os

import numpy as np

from JumpKing import JKGame


class JumpKingAgentStraightRay:

	def __init__(self, game):

		self.game = game

		self.ray_angles = [
			170,
			160,
			150,
			140,
			130,
			120,
			110,
			100,
			80,
			70,
			60,
			50,
			40,
			30,
			20,
			10
		]

		# 16 rayos × (hit + distancia)
		# + 3 ground rays

		self.observation_size = 35

	def reset(self):

		self.cached_ray_state = [0.0] * 32
		self.cached_ground_state = [0.0] * 3

	def get_state(self):

		state = []

		if self.game.move_available():

			ray_state = []

			x = self.game.king.rect.centerx
			y = self.game.king.rect.centery

			for angle in self.ray_angles:

				hit, distance = self.cast_straight_ray(
					x,
					y,
					angle
				)

				ray_state.append(hit)
				ray_state.append(distance)

			self.cached_ray_state = ray_state

			self.cached_ground_state = list(
				self.get_ground_sensors()
			)

		state.extend(
			self.cached_ray_state
		)

		state.extend(
			self.cached_ground_state
		)

		return np.array(
			state,
			dtype=np.float32
		)
	
	def cast_straight_ray(
		self,
		x,
		y,
		angle,
		max_distance=300,
		step_size=3
	):

		current_level = self.game.king.levels.current_level

		radians = math.radians(angle)

		dx = math.cos(radians)
		dy = -math.sin(radians)

		points = [(x, y)]

		prev_y = y

		for distance in range(0, max_distance, step_size):

			test_x = x + dx * distance
			test_y = y + dy * distance

			points.append(
				(test_x, test_y)
			)

			collision = self.game.find_collision(
				test_x,
				test_y,
				prev_y,
				current_level,
				360
			)

			prev_y = test_y

			if collision is None:
				continue

			if collision["collision_kind"] == "ceiling":

				if os.environ.get("render", "0") == "1":

					self.game.debug_rays.append(
						(points.copy(), "ceiling")
					)

				return (
					0.0,
					1.0 - distance / max_distance
				)

			elif collision["collision_kind"] == "wall":

				if os.environ.get("render", "0") == "1":

					self.game.debug_rays.append(
						(points.copy(), "wall")
					)

				return (
					1.0,
					1.0 - distance / max_distance
				)

		if os.environ.get("render", "0") == "1":

			self.game.debug_rays.append(
				(points.copy(), "none")
			)

		return (
			-1.0,
			0.0
		)
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