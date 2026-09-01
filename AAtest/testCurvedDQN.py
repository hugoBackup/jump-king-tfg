import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from envs.JumpkingcurvedRayEnv import JumpKingCurvedRayEnv

os.environ["render"] = "1"

#Script de evaluacion de los modelos CurvedDQN
# ============================================================
# CONFIGURACIÓN DEL AGENTE
# ============================================================

NUM_EPISODES = 100

MODEL_PATH = "resultadosFinales/jumpKingCurvedRayPunishDQN_4days"

AGENT_NAME = "DQN_curved_Punish"
ALGORITHM = "DQN"
OBSERVATION = "CurvedRays"
REWARD_FUNCTION = "Punish"


# ============================================================
# ARCHIVOS CSV
# ============================================================

RESULTS_CSV = "BBcsvResults/evaluation_resultsCurvedPPOPunishDQN.csv"
TRAJECTORIES_CSV = "BBcsvResults/evaluation_trajectoriesCurvedPPOPunishDQN.csv"


# ============================================================
# CREAR ENTORNO Y MODELO
# ============================================================

env = JumpKingCurvedRayEnv()

print(env.__dict__.keys())

model = DQN.load(
    MODEL_PATH,
    env=env
)


# ============================================================
# PREPARAR CSV DE RESULTADOS
# ============================================================

results_file_exists = os.path.exists(RESULTS_CSV)

results_file = open(
    RESULTS_CSV,
    mode="a",
    newline="",
    encoding="utf-8"
)

results_writer = csv.writer(results_file)


if not results_file_exists:

    results_writer.writerow([
        "agent",
        "algorithm",
        "observation",
        "reward_function",
        "episode",
        "total_reward",
        "max_height",
        "max_level",
        "actions_to_max_height"
    ])


# ============================================================
# PREPARAR CSV DE TRAYECTORIAS
# ============================================================

trajectories_file_exists = os.path.exists(TRAJECTORIES_CSV)

trajectories_file = open(
    TRAJECTORIES_CSV,
    mode="a",
    newline="",
    encoding="utf-8"
)

trajectories_writer = csv.writer(trajectories_file)


if not trajectories_file_exists:

    trajectories_writer.writerow([
        "agent",
        "algorithm",
        "observation",
        "reward_function",
        "episode",
        "step",
        "action",
        "y",
        "global_height",
        "reward",
        "cumulative_reward"
    ])


# ============================================================
# EVALUACIÓN
# ============================================================

episode_rewards = []
episode_max_heights = []
episode_max_levels = []
episode_actions_to_max_height = []


for episode in range(1, NUM_EPISODES + 1):

    obs, _ = env.reset()

    total_reward = 0
    max_height = float("-inf")
    max_height_action = 0
    max_level = 0
    action_count = 0

    print("\n")
    print("=" * 70)
    print(f"EPISODIO {episode}")
    print("=" * 70)

    while True:

        # ----------------------------------------------------
        # Selección de acción
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=False
        )

        # ----------------------------------------------------
        # Ejecutamos la acción
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(action)

        action_count += 1
        total_reward += reward

        # ----------------------------------------------------
        # Posición actual
        # ----------------------------------------------------

        y = env.game.king.y

        level = env.game.king.levels.current_level

        height = env.game.get_global_height(
            level,
            y
        )

        print(
            f"LEVEL={level} | "
            f"king.y={env.game.king.y} | "
            f"rect.top={env.game.king.rect.top} | "
            f"rect.bottom={env.game.king.rect.bottom} | "
            f"GLOBAL_HEIGHT={height}"
        )

        # ----------------------------------------------------
        # Actualizamos altura máxima
        # ----------------------------------------------------

        if height > max_height:

            max_height = height

            max_height_action = action_count

        # ----------------------------------------------------
        # Actualizamos nivel máximo
        # ----------------------------------------------------

        if level > max_level:

            max_level = level

        # ----------------------------------------------------
        # Guardar trayectoria en CSV
        # ----------------------------------------------------

        trajectories_writer.writerow([
            AGENT_NAME,
            ALGORITHM,
            OBSERVATION,
            REWARD_FUNCTION,
            episode,
            action_count,
            action,
            y,
            height,
            reward,
            total_reward
        ])

        # ----------------------------------------------------
        # Información por pantalla
        # ----------------------------------------------------

        print(
            f"Acción: {action} | "
            f"Y: {y} | "
            f"Altura global: {height} | "
            f"Reward: {reward:.4f} | "
            f"Reward acumulada: {total_reward:.4f}"
        )

        # ----------------------------------------------------
        # Fin del episodio
        # ----------------------------------------------------

        if terminated or truncated:

            break


    # ========================================================
    # RESULTADO DEL EPISODIO
    # ========================================================

    episode_rewards.append(total_reward)

    episode_max_heights.append(max_height)

    episode_max_levels.append(max_level)

    episode_actions_to_max_height.append(
        max_height_action
    )


    # ========================================================
    # GUARDAR RESULTADO EN CSV
    # ========================================================

    results_writer.writerow([
        AGENT_NAME,
        ALGORITHM,
        OBSERVATION,
        REWARD_FUNCTION,
        episode,
        total_reward,
        max_height,
        max_level,
        max_height_action
    ])

    # Forzamos que los datos se escriban inmediatamente
    results_file.flush()
    trajectories_file.flush()


    # ========================================================
    # RESUMEN DEL EPISODIO
    # ========================================================

    print("\n--- RESUMEN DEL EPISODIO ---")

    print(
        f"Recompensa total: "
        f"{total_reward:.4f}"
    )

    print(
        f"Altura máxima: "
        f"{max_height}"
    )

    print(
        f"Acciones hasta altura máxima: "
        f"{max_height_action}"
    )

    print(
        f"Nivel máximo: "
        f"{max_level}"
    )


# ============================================================
# RESULTADOS FINALES
# ============================================================

average_reward = (
    sum(episode_rewards)
    / NUM_EPISODES
)

average_height = (
    sum(episode_max_heights)
    / NUM_EPISODES
)

average_level = (
    sum(episode_max_levels)
    / NUM_EPISODES
)

average_actions = (
    sum(episode_actions_to_max_height)
    / NUM_EPISODES
)


# ============================================================
# CERRAR CSV
# ============================================================

results_file.close()

trajectories_file.close()


# ============================================================
# MOSTRAR RESULTADOS FINALES
# ============================================================

print("\n")
print("=" * 70)
print("RESULTADOS FINALES")
print("=" * 70)

print(
    f"Agente: "
    f"{AGENT_NAME}"
)

print(
    f"Recompensa media: "
    f"{average_reward:.4f}"
)

print(
    f"Altura máxima media: "
    f"{average_height:.2f}"
)

print(
    f"Nivel máximo medio: "
    f"{average_level:.2f}"
)

print(
    f"Acciones medias hasta altura máxima: "
    f"{average_actions:.2f}"
)


print("\nResultados individuales:")


for i in range(NUM_EPISODES):

    print(
        f"Episodio {i + 1}: "
        f"Reward={episode_rewards[i]:.4f} | "
        f"MaxHeight={episode_max_heights[i]} | "
        f"MaxLevel={episode_max_levels[i]} | "
        f"Acciones={episode_actions_to_max_height[i]}"
    )