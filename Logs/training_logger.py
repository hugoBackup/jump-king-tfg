import csv
import os
import time
from datetime import datetime

#guardado de metricas durante el entrenamiento
class TrainingLogger:

    def __init__(self, agent_name):

        os.makedirs("logs", exist_ok=True)

        self.agent_name = agent_name

        self.episode = 0

        self.best_height_ever = 0

        self.start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.csv_path = (
            f"logs/fullinfo/kissFullFinal/{agent_name}_{timestamp}DQN.csv"
        )

        self.file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "episode",
            "reward",
            "max_height",
            "max_level",
            "actions",
            "actions_to_max_height",
            "unique_positions",
            "best_height_ever",
            "time_seconds"
        ])

        self.file.flush()

    def log_episode(
        self,
        reward,
        max_height,
        max_level,
        actions,
        actions_to_max_height,
        unique_positions
    ):
        print("LOG EPISODIO", self.episode)
        self.episode += 1

        self.best_height_ever = max(
            self.best_height_ever,
            max_height
        )

        save = (
            self.episode % 10 == 0
            or max_height >= self.best_height_ever
        )

        if not save:
            return
        elapsed_time = time.time() - self.start_time
        self.writer.writerow([
            self.episode,
            reward,
            max_height,
            max_level,
            actions,
            actions_to_max_height,
            unique_positions,
            self.best_height_ever,
            elapsed_time
        ])

        self.file.flush()   

    def close(self):

        self.file.close()
        