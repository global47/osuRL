import numpy as np

from osuEnv import osuAimEnv
from testEnv import aim_action


def evaluate(policy, seeds, env=None, reset_options=None):
    if env is None:
        env = osuAimEnv(num_targets=1)

    total_hits = 0
    total_targets = 0
    total_bad_clicks = 0
    episode_rewards = []

    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed, options=reset_options)
            total_reward = 0.0

            while True:
                action = policy(observation, env)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            total_hits += info["hits"]
            total_targets += env.unwrapped.num_targets
            total_bad_clicks += info["bad_clicks"]
            episode_rewards.append(total_reward)
    finally:
        env.close()

    return {
        "hit_rate": total_hits / total_targets,
        "bad_clicks_per_episode": total_bad_clicks / len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)),
    }


if __name__ == "__main__":
    results = evaluate(aim_action, range(1000, 1100))

    for name, value in results.items():
        print(f"{name}: {value:.3f}")
