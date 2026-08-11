from stable_baselines3.common.callbacks import BaseCallback
import time


class TrainingMetricsCallback(BaseCallback):

    def __init__(
        self,
        metrics_logger,
        log_freq=1000,
        verbose=0
    ):

        super().__init__(verbose)

        self.metrics_logger = metrics_logger
        self.log_freq = log_freq

        self.last_log_timestep = 0


    def _on_step(self):

        current_timesteps = self.num_timesteps

        if (
            current_timesteps - self.last_log_timestep
            >= self.log_freq
        ):

            self._save_metrics()

            self.last_log_timestep = current_timesteps

        return True


    def _save_metrics(self):

        values = self.logger.name_to_value

        timesteps = self.num_timesteps

        # En tu entorno:
        # 100 timesteps = 1 episodio
        episode = int(timesteps / 100)

        ep_rew_mean = values.get(
            "rollout/ep_rew_mean",
            None
        )

        ep_len_mean = values.get(
            "rollout/ep_len_mean",
            None
        )

        elapsed_time = (
            time.time()
            - self.metrics_logger.start_time
        )


        # ====================================================
        # DQN
        # ====================================================

        if self.metrics_logger.algorithm == "DQN":

            loss = values.get(
                "train/loss",
                None
            )

            exploration_rate = values.get(
                "rollout/exploration_rate",
                None
            )

            self.metrics_logger.log([
                timesteps,
                episode,
                ep_rew_mean,
                ep_len_mean,
                loss,
                exploration_rate,
                elapsed_time
            ])


        # ====================================================
        # PPO
        # ====================================================

        elif self.metrics_logger.algorithm == "PPO":

            value_loss = values.get(
                "train/value_loss",
                None
            )

            policy_gradient_loss = values.get(
                "train/policy_gradient_loss",
                None
            )

            approx_kl = values.get(
                "train/approx_kl",
                None
            )

            entropy_loss = values.get(
                "train/entropy_loss",
                None
            )

            explained_variance = values.get(
                "train/explained_variance",
                None
            )

            self.metrics_logger.log([
                timesteps,
                episode,
                ep_rew_mean,
                ep_len_mean,
                value_loss,
                policy_gradient_loss,
                approx_kl,
                entropy_loss,
                explained_variance,
                elapsed_time
            ])