"""Run a simple aiming policy for visual debugging."""

import argparse
from pathlib import Path

import numpy as np

from osuEnv import osuAimEnv


def aim_action(observation, env):
    """Move toward the circle and click within its timing window."""
    dx, dy, distance, time_until = observation
    movement = np.zeros(2, dtype=np.float32)
    if distance > 0:
        speed_fraction = 1.0
        if time_until > 0:
            speed_fraction = np.clip(
                distance / (time_until / 1000.0) / env.cursor_speed, 0.0, 1.0
            )
        movement = np.array([dx, dy], dtype=np.float32) / distance * speed_fraction

    click = float(
        distance <= env.hit_radius and abs(time_until) <= env.hit_window
    )
    return np.array([*movement, click], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "beatmap",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name(
            "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu"
        ),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = osuAimEnv(
        render_mode=None if args.headless else "human",
        beatmap_path=None if args.synthetic else args.beatmap,
        num_targets=1
    )
    try:
        observation, _ = env.reset(seed=42)
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, info = env.step(
                aim_action(observation, env)
            )
            total_reward += reward
            if terminated or truncated:
                break
            if not args.headless and env.window is None:
                break
        print(f"Completed {info['target_index']} targets; total reward: {total_reward:.0f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
