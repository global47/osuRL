from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, TimingOnlyAction, NearbyStart


def main():
    output_dir = Path(__file__).parent / "runs" / "ppo_circle_accuracy_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        NormalizeObservation(
            NearbyStart(osuAimEnv(num_targets=1, bad_click_penalty=0.1 ),
                        distance=45.0
                             )),
        filename=str(output_dir / "training"),
        info_keywords=("hits", "misses", "bad_clicks"),
    )
    observation, _ = env.reset(seed=42)

    print("Action shape:", env.action_space.shape)
    print("Starting distance:", observation[2] * 640)
    print("Observation shape:", env.observation_space.shape)

    try:
        checkpoint = Path(__file__).parent / "runs" / "ppo_circle_accuracy_v1" / "model.zip"

        model = PPO.load(checkpoint, env=env, device="cpu")

        model.learn(total_timesteps=100_000, reset_num_timesteps=False)
        model.save(output_dir / "model")
    finally:
        env.close()


if __name__ == "__main__":
    main()
