import gymnasium as gym
from gymnasium import spaces
import numpy as np
from JumpKing import JKGame


class JumpKingEnvContinuous(gym.Env):

    def __init__(self, render_mode=False):

        super().__init__()
        self.render_mode = render_mode

        self.game = JKGame(max_step=1000)

        # acción continua: [dirección, potencia]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # observación: grid
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(40 * 40,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        _, state = self.game.reset()
        return np.array(state, dtype=np.float32), {}

    def execute_jump(self, action):

        direction = float(action[0])
        power = float(action[1])

        # clamp por seguridad
        direction = np.clip(direction, -1.0, 1.0)
        power = np.clip(power, 0.0, 1.0)

        max_charge = 30
        charge = int(power * max_charge)

        if direction < 0:
            hold = 3      # izquierda + espacio
            release = 1   # izquierda
        else:
            hold = 2      # derecha + espacio
            release = 0   # derecha

        # cargar salto
        for _ in range(charge):
            self.game.step(hold)

        # soltar salto
        self.game.step(release)

    def step(self, action):

        # estado inicial
        old_level = self.game.king.levels.current_level
        old_y = self.game.king.y

        old_height = self.game.get_global_height(old_level, old_y)

        # ejecutar salto completo
        self.execute_jump(action)

        # simular hasta que termine
        while not self.game.move_available():
            self.game.step(None)

        # estado final
        final_level = self.game.king.levels.current_level
        final_y = self.game.king.y

        new_height = self.game.get_global_height(final_level, final_y)

        # reward = cambio de altura
        reward = new_height - old_height

        # observación
        state = self.game.get_layout_grid().flatten()

        terminated = True   # episodio = 1 salto
        truncated = False

        return state, reward, terminated, truncated, {}

    def render(self):
        pass