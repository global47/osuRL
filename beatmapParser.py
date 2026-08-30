import numpy as np


def load_hit_circles(filepath):

    positions = []
    times = []

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

            # We'll add filtering here shortly

            positions.append([x, y])
            times.append(time)

    return (
        np.array(positions, dtype=np.float32),
        np.array(times, dtype=np.float32)
    )
