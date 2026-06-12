from JumpKingMulti import JumpKingMulti
from stable_baselines3 import PPO

print("Creando entorno...")
env = JumpKingMulti()

print("Cargando modelo...")
model = PPO.load(
    "JumpKingMatrix2.0",
    env=env
)

print("Modelo cargado correctamente")

print("\n=== Hiperparámetros ===")
print("learning_rate =", model.learning_rate)
print("n_steps =", model.n_steps)
print("batch_size =", model.batch_size)

obs, info = env.reset()

print("\n=== Probando acciones ===")

for i in range(20):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    print(
        f"Paso {i:02d} -> acción {int(action)}"
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    print(
        f"reward={reward:.3f}"
    )

    if terminated or truncated:

        print("Reset episodio")

        obs, info = env.reset()

print("\nPrueba finalizada")