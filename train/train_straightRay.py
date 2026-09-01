import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.jumpKingStraightRayEnv import JumpKingStraightRayEnv

from Logs.training_metrics_logger import TrainingMetricsLogger
from callbacks.training_metrics_callback import TrainingMetricsCallback


os.environ["render"] = "0"
type = "punish"
#Script de entrenamiento de los modelos StraightPPO
# ============================================================
# ENTORNO
# ============================================================

env = JumpKingStraightRayEnv()


# ============================================================
# MODELO PPO
# ============================================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64
)


# ============================================================
# LOGGER DE MÉTRICAS DE PPO
# ============================================================

metrics_logger = TrainingMetricsLogger(
    agent_name="jumpKingAgentStraightRay",
    algorithm="PPO"
)

metrics_callback = TrainingMetricsCallback(
    metrics_logger
)


# ============================================================
# CHECKPOINTS
# ============================================================

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=f"./checkpointStraightRay{type}",
    name_prefix=f"straightRay{type}"
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

MAX_TIME = 4 * 24 * 60 * 60  # 4 días

start_time = time.time()


while time.time() - start_time < MAX_TIME:

    model.learn(
        total_timesteps=10_000,
        callback=[
            checkpoint_callback,
            metrics_callback
        ],
        reset_num_timesteps=False
    )


# ============================================================
# GUARDADO FINAL
# ============================================================

model.save(
    f"jumpKingStraightRay{type}_4days"
)


# ============================================================
# CERRAR LOGGERS
# ============================================================

metrics_logger.close()
env.close()