import gymnasium as gym
from gymnasium import spaces
import numpy as np
from JumpKing import JKGame


class JumpKingEnvContinuous(gym.Env):

    def __init__(self, render_mode=False):

        super().__init__()
        self.render_mode = render_mode

        self.game = JKGame(max_step=1000)

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.game.reset()
        sample_state = self.game.get_state()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=sample_state.shape,
            dtype=np.float32
        )


    def reset(self, seed=None, options=None):

        state = self.game.reset()

        return np.array(state, dtype=np.float32), {}


    def step(self, action):

        old_level = self.game.king.levels.current_level
        old_y = self.game.king.y
        old_height = self.game.get_global_height(old_level, old_y)

        self.execute_jump(action)

        max_sim_steps = 300
        sim_steps = 0

        while not self.game.move_available() and sim_steps < max_sim_steps:
            self.game.step(None)
            sim_steps += 1

        final_level = self.game.king.levels.current_level
        final_y = self.game.king.y
        new_height = self.game.get_global_height(final_level, final_y)

        reward = np.clip((new_height - old_height) / 100.0, -1.0, 1.0)
        reward -= 0.01

        state = np.array(self.game.get_state(), dtype=np.float32)

        terminated = True
        truncated = False

        return state, reward, terminated, truncated, {}
    
    def execute_jump(self, action):

        direction = float(action[0])
        power = float(action[1])

        direction = np.clip(direction, -1.0, 1.0)
        power = np.clip(power, 0.0, 1.0)

        max_charge = 30
        charge = int(power * max_charge)

        if direction < 0:
            hold = 3
            release = 1
        else:
            hold = 2
            release = 0

        # cargar salto
        for _ in range(charge):
            self.game.step(hold)

        # soltar salto
        self.game.step(release)