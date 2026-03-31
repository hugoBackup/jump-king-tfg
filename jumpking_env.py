import gymnasium as gym
from gymnasium import spaces
import numpy as np
from JumpKing import JKGame


class JumpKingEnv(gym.Env):

    def __init__(self, render_mode=False):

        super().__init__()
        self.render_mode = render_mode

        self.game = JKGame(max_step=1000)

        self.action_space = spaces.Discrete(63)

        self.observation_space = spaces.Box(
            low=-10000,
            high=10000,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        done, state = self.game.reset()

        return np.array(state, dtype=np.float32), {}

    def step(self, action):

        # -------- NO OP --------
        if action == 0:
            game_action = None
            state, reward, done = self.game.step(game_action)

        # -------- WALK LEFT --------
        elif action == 1:
            game_action = 1
            state, reward, done = self.game.step(game_action)

        # -------- WALK RIGHT --------
        elif action == 2:
            game_action = 0
            state, reward, done = self.game.step(game_action)

        # -------- LEFT JUMPS --------
        elif 3 <= action <= 32:

            charge = action - 2

            # hold SPACE for charge frames
            for _ in range(charge):
                self.game.step(3)   # LEFT + SPACE

            # release jump
            state, reward, done = self.game.step(1)  # LEFT

        # -------- RIGHT JUMPS --------
        elif 33 <= action <= 62:

            charge = action - 32

            for _ in range(charge):
                self.game.step(2)   # RIGHT + SPACE

            state, reward, done = self.game.step(0)  # RIGHT

        obs = np.array(state, dtype=np.float32)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):

        pass