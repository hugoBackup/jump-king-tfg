class JumpPredictionResult:

	def __init__(self):

		self.valid_floor = False

		self.relative_height = 0.0

		self.relative_x = 0.0

		self.collision_type = "none"

		self.points = []

		self.hit_x = None
		self.hit_y = None

		self.hit_ceiling = False

		self.wall_bounces = 0

		self.landing_x = None
		self.landing_y = None
		self.landing_level = None

		self.global_height = None