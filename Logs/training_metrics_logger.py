import csv
import os
import time
from datetime import datetime


class TrainingMetricsLogger:

    def __init__(self, agent_name, algorithm):

        os.makedirs("logs", exist_ok=True)

        self.agent_name = agent_name
        self.algorithm = algorithm
        self.start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        self.csv_path = (
            f"logs/fullinfo/kissFullFinal/{agent_name}_{algorithm}_training_{timestamp}.csv"
        )

        self.file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        if algorithm == "DQN":

            self.writer.writerow([
                "timesteps",
                "episode",
                "ep_rew_mean",
                "ep_len_mean",
                "loss",
                "exploration_rate",
                "time_seconds"
            ])

        elif algorithm == "PPO":

            self.writer.writerow([
                "timesteps",
                "episode",
                "ep_rew_mean",
                "ep_len_mean",
                "value_loss",
                "policy_gradient_loss",
                "approx_kl",
                "entropy_loss",
                "explained_variance",
                "time_seconds"
            ])

        self.file.flush()

    def log(self, values):

        self.writer.writerow(values)

        self.file.flush()

    def close(self):

        self.file.close()