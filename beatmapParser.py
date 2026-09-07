import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    object_type: int


def load_hit_circles(filepath):

    hit_objects = []
    in_hit_objects = False

    with open(filepath, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line == "[HitObjects]":
                in_hit_objects = True
                continue

            if not in_hit_objects:
                continue

            if not line:
                continue

            parts = line.split(",")

            x = int(parts[0])
            y = int(parts[1])
            time = int(parts[2])
            object_type = int(parts[3])

            # Only include hit circles
            if object_type & 1:
                hit_objects.append(
                    HitObject(x=x, y=y, time=time, object_type=object_type)
                )
    return hit_objects


def main():

    parser = argparse.ArgumentParser(description="osu! RL beatmap parser")

    parser.add_argument(
        "osuFile",
        nargs="?",
        default="FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu",
        help="Path to an .osu file",
    )

    args = parser.parse_args()

    positions, times = load_hit_circles(args.osuFile)

    print("File:", args.osuFile)
    print("Number of circles:", len(positions))
    print()

    print(f"Clrcle 0: position={positions[0]}, time={times[0]:.0f} ms")

    for i in range(1, len(times)):
        previousPos = positions[i - 1]
        previousTime = times[i - 1]

        diffPos = positions[i] - previousPos
        diffTime = times[i] - previousTime
        jump_distance = np.linalg.norm(diffPos)

        print(
            f"Circle {i}: position={positions[i]}, time={times[i]:.0f} ms | Positional difference={jump_distance}, Temporal Difference ={diffTime:.0f} ms"
        )


if __name__ == "__main__":
    main()
