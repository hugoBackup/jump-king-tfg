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

        self.current_step = 0
        self.total_reward = 0.0

        return np.array(state, dtype=np.float32), {}


    def step(self, action):

        old_level = self.game.king.levels.current_level
        old_y = self.game.king.y
        old_height = self.game.get_global_height(old_level, old_y)

        if not hasattr(self, "success_streak"):
            self.success_streak = 0
        if not hasattr(self, "jumps_in_level"):
            self.jumps_in_level = 0
        if not hasattr(self, "last_level"):
            self.last_level = old_level
        if not hasattr(self, "total_reward"):
            self.total_reward = 0.0
        if not hasattr(self, "successful_jumps"):
            self.successful_jumps = 0
        if not hasattr(self, "failed_jumps"):
            self.failed_jumps = 0
        if not hasattr(self, "neutral_jumps"):
            self.neutral_jumps = 0
        if not hasattr(self, "jump_counter"):
            self.jump_counter = 0
        if not hasattr(self, "numero_acciones"):
            self.numero_acciones = 0

        self.execute_jump(action)

        max_sim_steps = 200
        sim_steps = 0

        while not self.game.move_available() and sim_steps < max_sim_steps:
            self.game.step(None)
            sim_steps += 1

        final_level = self.game.king.levels.current_level
        final_y = self.game.king.y
        new_height = self.game.get_global_height(final_level, final_y)

        reward = 0.0

        if final_level != self.last_level:
            self.jumps_in_level = 0
            self.last_level = final_level

        self.jumps_in_level += 1

        if self.jumps_in_level > 30:
            reward -= 5.0
            self.jumps_in_level = 0
            self.success_streak = 0

        delta = new_height - old_height

        jump_success = False
        jump_fail = False

        if delta > 5:
            self.success_streak += 1
            reward += self.get_fibonacci_reward(self.success_streak)
            self.successful_jumps += 1
            jump_success = True

        elif delta >= -5:
            reward += 0.02
            self.neutral_jumps += 1
            self.success_streak = 0

        else:
            reward -= 0.2
            self.failed_jumps += 1
            self.success_streak = 0
            jump_fail = True

        state = np.array(self.game.get_state(), dtype=np.float32)

        self.current_step += 1
        self.total_reward += reward
        self.jump_counter += 1
        self.numero_acciones += 1

        #if jump_success:
            #print(f"✔️ Acción {self.numero_acciones} | Jump {self.jump_counter} | Reward: {reward:.2f}")
        #elif jump_fail:
            #print(f"❌ Acción {self.numero_acciones} | Jump {self.jump_counter} | Reward: {reward:.2f}")

        if self.jump_counter % 1 == 0:
            print(
                f"Resumen -> Acciones: {self.numero_acciones} | ✔️ {self.successful_jumps} | ❌ {self.failed_jumps} | "
                f"~ {self.neutral_jumps} | Total Reward: {self.total_reward:.2f}"
            )

            self.successful_jumps = 0
            self.failed_jumps = 0
            self.neutral_jumps = 0

        truncated = self.current_step >= 1000
        terminated = False

        return state, reward, terminated, truncated, {}
    
    def get_fibonacci_reward(self, n):
        if n <= 0:
            return 0
        if n == 1:
            return 1
        a, b = 1, 2
        for _ in range(n - 1):
            a, b = b, a + b
        return a
    
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