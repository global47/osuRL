import argparse
from dataclasses import dataclass
from math import hypot
from pathlib import Path


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    object_type: int


def load_hit_circles(filepath):
    """Read circle objects in file order, preserving their original timestamps."""

    hit_objects = []
    in_hit_objects = False

    with open(filepath, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line == "[HitObjects]":
                in_hit_objects = True
                continue

            if line.startswith("["):
                in_hit_objects = False
                continue

            if not in_hit_objects:
                continue

            if not line or line.startswith("//"):
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
        default=Path(__file__).with_name(
            "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu"
        ),
        help="Path to an .osu file",
    )

    args = parser.parse_args()

    hit_objects = load_hit_circles(args.osuFile)

    print("File:", args.osuFile)
    print("Number of circles:", len(hit_objects))
    print()

    if not hit_objects:
        return

    print(
        f"Circle 0: position=({hit_objects[0].x}, {hit_objects[0].y}), "
        f"time={hit_objects[0].time:.0f} ms"
    )

    for i in range(1, len(hit_objects)):
        previous = hit_objects[i - 1]
        current = hit_objects[i]

        diff_time = current.time - previous.time
        jump_distance = hypot(current.x - previous.x, current.y - previous.y)

        print(
            f"Circle {i}: position=({current.x}, {current.y}), "
            f"time={current.time:.0f} ms | Positional difference={jump_distance}, "
            f"Temporal Difference={diff_time:.0f} ms"
        )


if __name__ == "__main__":
    main()
