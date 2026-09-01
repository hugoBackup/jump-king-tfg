import os
import math
import csv
import time

import numpy as np

from SearchNode import SearchNode
from JumpKing import JumpPredictionResult
from JumpKing import JKGame
#esta clase es utilizada por el modelo basado en reglas que se utilia para comparar resultados con los modelos de RL
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

		self.game = game
		

		self.current_action = None
		self.current_jump_power = None
		self.action_frame = 0

		self.search_depth = 3
		self.max_branches = 10

		self.tree_debug_rays = []
		self.show_tree = False

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

		self.jump_counter = 0
		self.execution_action_counter = 0

		self.execution_max_height = float("-inf")
		self.execution_max_height_actions = 0


		self.results_file = open(
			"ZZZZresultsTreeSearch.csv",
			mode="w",
			newline="",
			encoding="utf-8"
		)

		self.results_writer = csv.writer(self.results_file)

		self.results_writer.writerow([
			"action",
			"level",
			"height",
			"max_height",
			"actions_to_max_height"
		])

		self.results_file.flush()

		self.log_file = open(
			"treeSearchLog.csv",
			"w",
			newline="",
			encoding="utf-8"
		)

		self.writer = csv.writer(self.log_file)

		self.writer.writerow([
			"jump",
			"height",
			"level"
		])

	def reset(self):
		pass

	def get_state(self):

		return np.zeros(1, dtype=np.float32)

	
	#Genera la trayectoria de lso rayos curvos	
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

		x = math.sin(angle) * speed
		y = math.cos(angle) * speed

		angle = math.pi/2 - math.atan2(y, x)
		length = math.hypot(x, y)

		# convertir a velocidades cartesianas
		vx = math.sin(angle) * length
		vy = -math.cos(angle) * length

		return vx, vy

	
	#toma los detayes sobre las caracteristicas resultantes de cada rayo. Como el punto de caida , altura , etc.
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
			not self.game.tree_move_available()
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

					self.tree_debug_rays.append(
						(points.copy(), "none")
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
				and step < 5
			):

				collision_kind = "wall"

			if collision_kind == "floor":

				if self.is_transition_false_floor(
					collision,
					current_level
				):
					continue
				

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

					self.tree_debug_rays.append(
						(
							points.copy(),
							"floor"
						)
					)
				result.collision_type = "floor"

				return result

			elif collision_kind == "ceiling":

				if not has_hit_ceiling:

					if os.environ.get("render", "0") == "1":

						self.tree_debug_rays.append(
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

						self.tree_debug_rays.append(
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

						self.tree_debug_rays.append(
							(points.copy(), "wall")
						)

					return result

		if os.environ.get("render", "0") == "1":

			self.tree_debug_rays.append(
				(points.copy(), "none")
			)

		return result

	# Genera los nodos hijos probando todas las acciones de salto posibles
	def expand_node(self, node):

		children = []

		for action, (jump_count, direction) in enumerate(self.actions):

			# ==========================================
			# PREDICCIÓN DEL RAYO
			# ==========================================

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

			# La potencia REAL que se ejecutará.
			jump_power = jump_count

			# ==========================================
			# J30 -> EJECUTAR CON 31
			#
			# El rayo sigue siendo J30.
			# ==========================================

			if (
				jump_count == 30
				and node.level > 1
				and result.landing_level > node.level
			):
				
				jump_power = 31

			child = SearchNode(
				x=result.landing_x,
				y=result.landing_y,
				level=result.landing_level,
				global_height=result.global_height,
				action=action,
				jump_power=jump_power,
				direction=direction,
				parent=node
			)

			if result.landing_level > node.level:
				child.finished = True

			node.children.append(child)
			children.append(child)

		return children
	# Construye el árbol de búsqueda y selecciona la accion que lleva al mejor resultado del arbol
	def choose_action(self):

		self.tree_debug_rays.clear()

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
			self.search_depth,
			debug=True
		)

		best_leaf = self.get_best_leaf(root)

		path = self.get_path(best_leaf)

		if len(path) == 0:
			return

		self.current_action = path[0].action
		self.action_frame = 0

		self.jump_counter += 1

		height = self.game.get_global_height(
			self.game.king.levels.current_level,
			self.game.king.y
		)

		self.writer.writerow([
			self.jump_counter,
			height,
			self.game.king.levels.current_level,
			path[0].jump_power,
			path[0].direction
		])

		self.log_file.flush()

	
	#Explora recursivamente el arbol entero.
	def search(
		self,
		node,
		depth,
		debug=False
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
				depth - 1,
				debug=False
			)
	#busca la hoja del arbol que presenta mas altura global
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
	
	#reconstruye ña seciemcoa de accopmes desde ña raoz jasta ima hoja
	def get_path(self, leaf):

		path = []

		node = leaf

		while node.parent is not None:

			path.append(node)

			node = node.parent

		path.reverse()

		return path	

	#controla la ejecucion del salto elegido
	def get_action(self):

		if self.current_action is None:
			self.choose_action()

			# Potencia real que se utilizará para ejecutar
			self.current_jump_power = None

		if self.current_action is None:
			self.show_tree = False
			return None

		jump_count, direction = self.actions[self.current_action]

		# ==========================================
		# DETERMINAR POTENCIA REAL DEL SALTO
		# ==========================================

		if self.current_jump_power is None:

			self.current_jump_power = jump_count

			# Solo J30 puede convertirse en J31
			if jump_count == 30:

				x = self.game.king.rect.centerx
				y = self.game.king.rect.bottom - 10
				level = self.game.king.levels.current_level

				# IMPORTANTE:
				# la predicción sigue siendo J30
				vx, vy = self.get_ray_jump_vector(
					30,
					direction
				)

				result = self.evaluate_jump(
					x,
					y,
					level,
					30,
					direction,
					vx,
					vy
				)

				# Si el J30 predice que llegamos al siguiente nivel,
				# ejecutamos físicamente el salto con 31.
				if (
					result.valid_floor
					and result.landing_level > level
					and level > 1
				):
					self.current_jump_power = 31

					

		level = self.game.king.levels.current_level

		if level > 0:
			actual_jump_count = self.current_jump_power + 1
		else:
			actual_jump_count = self.current_jump_power


		hold_action = 3 if direction == "left" else 2
		release_action = 1 if direction == "left" else 0

		# ==========================================
		# FASE DE CARGA
		# ==========================================

		if self.action_frame < actual_jump_count:

			# Mientras se carga el salto,
			# mostramos el árbol que acaba de calcularse.
			self.show_tree = True

			self.action_frame += 1

			return hold_action

		# ==========================================
		# LIBERACIÓN DEL SALTO
		# ==========================================

		elif self.action_frame == actual_jump_count:

			# El salto comienza.
			# Dejamos de mostrar el árbol.
			self.show_tree = False

			self.action_frame += 1

			# ==========================================
			# INFORMACIÓN DE EJECUCIÓN
			# ==========================================

			self.execution_action_counter += 1

			level = self.game.king.levels.current_level
			y = self.game.king.y

			height = self.game.get_global_height(
				level,
				y
			)

			if height > self.execution_max_height:

				self.execution_max_height = height

				self.execution_max_height_actions = (
					self.execution_action_counter
				)

			self.results_writer.writerow([
				self.execution_action_counter,
				level,
				height,
				self.execution_max_height,
				self.execution_max_height_actions
			])

			self.results_file.flush()

			return release_action

		# ==========================================
		# FIN DE LA ACCIÓN
		# ==========================================

		else:

			self.current_action = None
			self.current_jump_power = None
			self.action_frame = 0

			return None


	# Comprueba si una colisión con el suelo pertenece realmente a una plataforma de transición
	def is_transition_false_floor(
		self,
		collision,
		current_level,
		tolerance=10
	):
		if collision is None:
			return False

		# Solo nos interesa una colisión clasificada como suelo
		if collision["collision_kind"] != "floor":
			return False

		# Solo el suelo perteneciente al nivel actual.
		# Si pertenece al siguiente nivel, NO lo descartamos.
		if collision["level"] != current_level:
			return False

		surface_y = collision["surface_y"]

		if surface_y is None:
			return False

		# La parte superior del nivel actual es y = 0.
		# Usamos un pequeño margen para evitar problemas de precisión.
		if abs(surface_y) <= tolerance:
			return True

		return False