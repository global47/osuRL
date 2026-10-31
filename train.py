"""Train a fresh PPO MLP and save configuration plus held-out seed evaluation."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from evaluate import evaluate
from osuEnv import osuAimEnv
from wrappers import NormalizeObservation, NearbyStart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["circles", "mixed"], default="circles")
    parser.add_argument("--beatmap", type=Path)
    parser.add_argument("--num-targets", type=int, default=1)
    parser.add_argument("--sim-hz", type=float, default=60)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--start-distance", type=float, default=45)
    parser.add_argument("--bad-click-penalty", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now().strftime("ppo_%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    if min(args.timesteps, args.eval_episodes, args.num_targets) < 1 or min(args.n_steps, args.batch_size) < 2:
        parser.error("Counts must be positive; n-steps and batch-size must be at least 2")
    if args.batch_size > args.n_steps or args.n_steps % args.batch_size:
        parser.error("batch-size must divide n-steps")
    kwargs = dict(object_mode=args.mode, beatmap_path=args.beatmap, num_targets=args.num_targets,
                  sim_hz=args.sim_hz, bad_click_penalty=args.bad_click_penalty)
    base = osuAimEnv(**kwargs)
    # Validate scenario and curriculum before creating output files.
    base.reset(seed=args.seed, options={"start_distance": args.start_distance})
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["interface_version"] = base.interface_version
    config["evaluation_start"] = "unrestricted; independent scenario seeds, no nearby-start wrapper"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = Monitor(NormalizeObservation(NearbyStart(base, distance=args.start_distance)),
                  filename=str(args.output / "training"), info_keywords=("hits", "misses", "bad_clicks"))
    try:
        model = PPO("MlpPolicy", env, seed=args.seed, n_steps=args.n_steps,
                    batch_size=args.batch_size, device="cpu", verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output / "model")
        def policy(observation, env):
            return model.predict(observation, deterministic=True)[0]
        results = evaluate(policy, range(args.eval_seed, args.eval_seed + args.eval_episodes),
                           NormalizeObservation(osuAimEnv(**kwargs)))
        results["training_timesteps"] = model.num_timesteps
        (args.output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved model, configuration and evaluation to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
