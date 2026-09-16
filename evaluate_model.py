from pathlib import Path

from stable_baselines3 import PPO

from evaluate import evaluate
from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, NearbyStart, TimingOnlyAction


def main():
    model_path = (
        Path(__file__).parent
        / "runs"
        / "ppo_nearby_45_soft_click"
        / "model.zip"
    )
    model = PPO.load(model_path, device="cpu")
    """
    env = NormalizeObservation(
        TimingOnlyAction(
            osuAimEnv(num_targets=1, start_on_target=True)
        )
    )
    """

    def policy(observation, env):
        action, _ = model.predict(observation, deterministic=True)
        return action
    """
    for target_time in [500.0, 1000.0, 1500.0, 2000.0]:
        env = NormalizeObservation(
                TimingOnlyAction(
                    osuAimEnv(num_targets=1, start_on_target=True)
                    )
                )
        
        results = evaluate(policy, seeds=[1000], env=env, reset_options={"first_target_time": target_time},)

        print(f"Target at {target_time:.0f} ms: {results}")
    """

    beatmap_path = Path(__file__).parent / (
            "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu"
            )

    env = NormalizeObservation(
            osuAimEnv(beatmap_path=beatmap_path, bad_click_penalty=0.1),
            )
        
    results = evaluate(
            policy,
            seeds=range(1000, 1100),
            env=env
            )
    print(f"random starting pos: {results}")



if __name__ == "__main__":
    main()
