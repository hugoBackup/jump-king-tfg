import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.JumpkingcurvedRayEnv import JumpKingCurvedRayEnv

from Logs.training_metrics_logger import TrainingMetricsLogger
from callbacks.training_metrics_callback import TrainingMetricsCallback


os.environ["render"] = "0"
type = "Kiss"


# ============================================================
# ENTORNO
# ============================================================

env = JumpKingCurvedRayEnv()


# ============================================================
# MODELO DQN
# ============================================================

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000
)


# ============================================================
# LOGGER DE MÉTRICAS DE DQN
# ============================================================

metrics_logger = TrainingMetricsLogger(
    agent_name="jumpKingAgentCurvedRay",
    algorithm="DQN"
)

metrics_callback = TrainingMetricsCallback(
    metrics_logger
)


# ============================================================
# CHECKPOINTS
# ============================================================

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=f"./checkpointCurvedRay{type}DQNFinal",
    name_prefix=f"curvedRay{type}DQNFinal"
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

MAX_TIME = 4 * 24 * 60 * 60

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
    f"jumpKingCurvedRay{type}DQN_4daysFinal"
)


# ============================================================
# CERRAR LOGGERS
# ============================================================

metrics_logger.close()
env.close()