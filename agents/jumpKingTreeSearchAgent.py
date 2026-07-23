import os
import math

import numpy as np

from SearchNode import SearchNode
from JumpKing import JumpPredictionResult
from JumpKing import JKGame

class jumpKingTreeSearchAgent:

	def __init__(self, game):

		self.actions = [
			(5, "left"),
			(5, "right"),
			(10, "left"),
			(10, "right"),
			(20, "left"),
			(20, "right"),
			(25, "left"),
			(25, "right"),
			(30, "left"),
			(30, "right")
		]
		#0 -> 5L
		#1 -> 5R
		#2 -> 10L
		#3 -> 10R
		#4 -> 20L
		#5 -> 20R
		#6 -> 25L
		#7 -> 25R
		#8 -> 30L
		#9 -> 30R

		self.game = game

		self.search_depth = 2
		self.max_branches = 10

		self.jump_counts = [
			5,
			10,
			20,
			25,
			30
		]

		self.directions = [
			"left",
			"right"
		]

	def reset(self):
		pass

	def get_state(self):

		self.choose_action()

		return np.zeros(1, dtype=np.float32)

	
		
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
			level,
			jump_count,
			direction,
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

		current_level = level
		level_height = 360

		

		points = [(x, y)]
		result = JumpPredictionResult()

		origin_y = y
		origin_x = x
		has_bounced = False
		has_hit_ceiling = False

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
			# El rayo ha alcanzado el siguiente nivel
			if collision is not None and collision["level"] > current_level:

				result.reaches_next_level = True


			collision_kind = collision["collision_kind"]
			rect = collision["rect"]
			local_py = collision["local_py"]

			if (
				collision_kind == "floor"
				and vy < 0
			):
				collision_kind = "wall"

			if collision_kind == "floor":
				

				result.collision_type = "floor"

				surface_y = collision["surface_y"]

				level_diff = collision["level"] - current_level
				surface_y_local = surface_y - (level_diff * 360)

				height_difference = origin_y - surface_y_local

				relative_height = (
					origin_y - surface_y_local
				) / 360.0

				height_difference = origin_y - surface_y_local

				if height_difference >= -20:

					result.landing_x = x
					result.landing_y = surface_y - 10
					result.landing_level = collision["level"]

					result.global_height = self.game.get_global_height(
						collision["level"],
						surface_y
					)

					result.valid_floor = True
					result.relative_height = min(
						max(height_difference / 360.0, 0.0),
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
				result.collision_type = "floor"

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

	def expand_node(self, node):

		children = []

		for action, (jump_count, direction) in enumerate(self.actions):

			vx, vy = self.get_ray_jump_vector(
				jump_count,
				direction
			)

			result = self.evaluate_jump(
				node.x,
				node.y,
				node.level,
				jump_count,
				direction,
				vx,
				vy
			)

			if not result.valid_floor:
				continue

			child = SearchNode(
				x=result.landing_x,
				y=result.landing_y,
				level=result.landing_level,
				global_height=result.global_height,
				action=action,
				jump_power=jump_count,
				direction=direction,
				parent=node
			)

			if result.landing_level > node.level:
				child.finished = True

			node.children.append(child)
			children.append(child)

		return children

	def choose_action(self):
	
			x = self.game.king.rect.centerx
			y = self.game.king.rect.bottom - 10
	
			root = SearchNode(
				x=x,
				y=y,
				level=self.game.king.levels.current_level,
				global_height=self.game.get_global_height(
					self.game.king.levels.current_level,
					y
				)
			)
	
			self.search(
				root,
				self.search_depth
			)
	
			
	
			best_leaf = self.get_best_leaf(root)

			path = self.get_path(best_leaf)

			if len(path) == 0:
				return

			action = path[0].action

			self.execute_action(action)

	

	def search(
		self,
		node,
		depth
	):

		if depth == 0:
			return

		if getattr(node, "finished", False):
			return

		children = self.expand_node(node)

		if len(children) == 0:
			return

		children.sort(
			key=lambda child: (
				child.global_height,
				child.jump_power
			),
			reverse=True
		)

		children = children[:self.max_branches]

		node.children = children

		for child in children:

			self.search(
				child,
				depth - 1
			)

	def get_best_leaf(self, node):

		if len(node.children) == 0:
			return node

		best_leaf = None

		for child in node.children:

			leaf = self.get_best_leaf(child)

			if (
				best_leaf is None
				or leaf.global_height > best_leaf.global_height
				or (
					leaf.global_height == best_leaf.global_height
					and leaf.jump_power > best_leaf.jump_power
				)
			):
				best_leaf = leaf

		return best_leaf	

	def get_path(self, leaf):

		path = []

		node = leaf

		while node.parent is not None:

			path.append(node)

			node = node.parent

		path.reverse()

		return path	

	def execute_action(self, action):

		jump_count, direction = self.actions[action]

		hold_action = 3 if direction == "left" else 2
		release_action = 1 if direction == "left" else 0

		for _ in range(jump_count):
			self.game.step(hold_action)

		self.game.step(release_action)

		