import gymnasium as gym
from gymnasium import spaces
import numpy as np
from JumpKing import JKGame


class JumpKingEnvDQN(gym.Env):

    def __init__(self, render_mode=False):

        super().__init__()
        self.render_mode = render_mode

        self.game = JKGame(max_step=1000)

        # 🔥 27 acciones
        self.action_space = spaces.Discrete(27)

        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(40 * 40,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        done, state = self.game.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):

        # -------- NO OP --------
        if action == 0:
            state, reward, done = self.game.step(None)

        # -------- WALK LEFT --------
        elif action == 1:
            state, reward, done = self.game.step(1)

        # -------- WALK RIGHT --------
        elif action == 2:
            state, reward, done = self.game.step(0)

        # -------- LEFT JUMPS --------
        elif 3 <= action <= 14:

            level = action - 2
            charge = int((level ** 1.5) * 2)

            for _ in range(charge):
                self.game.step(3)

            state, reward, done = self.game.step(1)

            # 🔥 ESPERAR A QUE TERMINE EL SALTO
            while not self.game.move_available():
                state, reward, done = self.game.step(None)

        # -------- RIGHT JUMPS --------
        elif 15 <= action <= 26:

            level = action - 14
            charge = int((level ** 1.5) * 2)

            for _ in range(charge):
                self.game.step(2)

            state, reward, done = self.game.step(0)

            # 🔥 ESPERAR A QUE TERMINE EL SALTO
            while not self.game.move_available():
                state, reward, done = self.game.step(None)

        obs = np.array(state, dtype=np.float32)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass