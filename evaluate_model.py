from pathlib import Path

from stable_baselines3 import PPO

from evaluate import evaluate
from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, NearbyStart, TimingOnlyAction


def main():
    model_path = (
        Path(__file__).parent
        / "runs"
        / "ppo_circle_accuracy_v2"
        / "model.zip"
    )
    model = PPO.load(model_path, device="cpu")

    def policy(observation, env):
        action, _ = model.predict(observation, deterministic=True)
        return action

    beatmap_path = Path(__file__).parent / (
            "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu"
            )

    for sim_hz in [60, 120, 240]:
        env = NormalizeObservation(
            osuAimEnv(beatmap_path=beatmap_path, bad_click_penalty=0.1, sim_hz=sim_hz),
        )
        
        results = evaluate(
                policy,
                seeds=range(5000, 5010),
                env=env
                )

        print(
            f"{sim_hz} Hz: "
            f"hit_rate={results['hit_rate']:.3f}, "
            f"accuracy={results['accuracy']:.3f}, "
            f"bad_clicks={results['bad_clicks_per_episode']:.3f}"
    )



if __name__ == "__main__":
    main()
