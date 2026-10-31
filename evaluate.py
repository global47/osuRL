
import argparse
import json
from pathlib import Path

import numpy as np

from osuEnv import osuAimEnv
from policies import scripted_action
from wrappers import NormalizeObservation


def evaluate(policy, seeds, env=None, reset_options=None):
    if env is None:
        env = osuAimEnv(num_targets=1)
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    counts = {300: 0, 100: 0, 50: 0, 0: 0}
    by_type = {kind: {"hits": 0, "misses": 0} for kind in ("circle", "slider", "spinner")}
    episodes = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed, options=reset_options)
            total_reward = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(policy(observation, env))
                total_reward += reward
                if terminated or truncated:
                    break
            for judgement, count in info["judgement_counts"].items():
                counts[judgement] += count
            for kind, metrics in info["object_metrics"].items():
                for key, value in metrics.items():
                    by_type[kind][key] += value
            episodes.append({"seed": int(seed), "targets": env.unwrapped.num_targets,
                             "hits": info["hits"], "misses": info["misses"],
                             "bad_clicks": info["bad_clicks"], "reward": total_reward})
    finally:
        env.close()
    total = sum(counts.values())
    hits = sum(row["hits"] for row in episodes)
    bad_clicks = sum(row["bad_clicks"] for row in episodes)
    for metrics in by_type.values():
        n = metrics["hits"] + metrics["misses"]
        metrics["hit_rate"] = metrics["hits"] / n if n else None
    return {"interface_version": env.unwrapped.interface_version,
            "targets": total, "hits": hits, "misses": counts[0], "bad_clicks": bad_clicks,
            "hit_rate": hits / total, "bad_clicks_per_episode": bad_clicks / len(episodes),
            "mean_reward": float(np.mean([row["reward"] for row in episodes])),
            "accuracy": sum(j * n for j, n in counts.items()) / (300 * total),
            "judgement_counts": counts, "by_type": by_type, "episodes": episodes}


def main(require_model=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=require_model)
    parser.add_argument("--mode", choices=["circles", "mixed"])
    parser.add_argument("--beatmap", type=Path)
    parser.add_argument("--num-targets", type=int)
    parser.add_argument("--sim-hz", type=float)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = {}
    if args.model:
        config_path = args.model.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
    mode = args.mode or config.get("mode", "mixed")
    env = osuAimEnv(object_mode=mode, beatmap_path=args.beatmap or config.get("beatmap"),
                    num_targets=args.num_targets if args.num_targets is not None else config.get("num_targets", 9),
                    sim_hz=args.sim_hz if args.sim_hz is not None else config.get("sim_hz", 60),
                    bad_click_penalty=config.get("bad_click_penalty", 0.1))
    policy = scripted_action
    if args.model:
        from stable_baselines3 import PPO
        env = NormalizeObservation(env)
        model = PPO.load(args.model, device="cpu")
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            parser.error("Checkpoint interface does not match this environment; select its mode or train a new model")
        def policy(observation, env):
            return model.predict(observation, deterministic=True)[0]
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    results = evaluate(policy, range(args.seed, args.seed + args.episodes), env)
    results["policy"] = str(args.model) if args.model else "scripted (privileged simulator state)"
    results["configuration"] = {"mode": mode, "beatmap": str(env.unwrapped.beatmap_path) if env.unwrapped.beatmap_path else None,
                                "sim_hz": env.unwrapped.sim_hz, "num_targets": env.unwrapped.num_targets}
    output = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
