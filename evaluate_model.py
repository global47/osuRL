from pathlib import Path

from stable_baselines3 import PPO

from evaluate import evaluate
from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, TimingOnlyAction


def main():
    model_path = (
        Path(__file__).parent
        / "runs"
        / "ppo_stationary_timing"
        / "model.zip"
    )
    model = PPO.load(model_path, device="cpu")

    env = NormalizeObservation(
        TimingOnlyAction(
            osuAimEnv(num_targets=1, start_on_target=True)
        )
    )

    def policy(observation, env):
        action, _ = model.predict(observation, deterministic=True)
        return action

    for target_time in [500.0, 1000.0, 1500.0, 2000.0]:
        env = NormalizeObservation(
                TimingOnlyAction(
                    osuAimEnv(num_targets=1, start_on_target=True)
                    )
                )
        
        results = evaluate(policy, seeds=[1000], env=env, reset_options={"first_target_time": target_time},)

        print(f"Target at {target_time:.0f} ms: {results}")

    #results = evaluate(policy, range(1000, 1100), env=env)

    for name, value in results.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
