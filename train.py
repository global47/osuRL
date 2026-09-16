from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, TimingOnlyAction


def main():
    output_dir = Path(__file__).parent / "runs" / "ppo_stationary_timing"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        NormalizeObservation(
            TimingOnlyAction(osuAimEnv(num_targets=1, start_on_target=True)
                             )),
        filename=str(output_dir / "training"),
        info_keywords=("hits", "misses", "bad_clicks"),
    )

    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=1,
            seed=42,
            device="cpu",
        )

        model.learn(total_timesteps=100_000)
        model.save(output_dir / "model")
    finally:
        env.close()


if __name__ == "__main__":
    main()
