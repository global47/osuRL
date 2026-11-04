import argparse
from dataclasses import dataclass, field
from math import hypot, isfinite
from pathlib import Path

from sliderPath import SliderPath


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    object_type: int
    duration: float = 0.0
    end_position: tuple[float, float] | None = None
    spans: int = 1
    path: SliderPath | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind == "slider":
            if not isfinite(self.duration) or self.duration <= 0 or self.spans < 1:
                raise ValueError("Slider requires positive duration and spans")
            if self.path is None:
                if self.end_position is None:
                    raise ValueError("Slider requires a path or endpoint")
                self.path = SliderPath.build("L", [(self.x, self.y), self.end_position],
                                             hypot(self.end_position[0] - self.x, self.end_position[1] - self.y))
            if self.path.points[0] != (self.x, self.y):
                raise ValueError("Slider path must begin at the object's position")
            self.end_position = self.path.points[-1]

    @property
    def kind(self):
        return {1: "circle", 2: "slider", 8: "spinner"}[self.object_type & 139]

    @property
    def end_time(self):
        return self.time + self.duration

    def position_at(self, time):
        #Distance based slider position, including repeats and the final endpoint.
        if self.kind != "slider":
            return self.x, self.y
        progress = min(max((time - self.time) / self.duration, 0.0), 1.0) * self.spans
        span = min(int(progress), self.spans - 1)
        fraction = progress - span
        if span % 2:
            fraction = 1 - fraction
        return self.path.position_at(fraction)

@dataclass
class BeatmapDifficulty:
    hp: float
    cs: float
    od: float
    ar: float

def load_difficulty(filepath):
    hp = cs = od = ar = None
    in_difficulty = False

    with open(filepath, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("//"):
                continue

            if line.startswith("["):
                in_difficulty = line == "[Difficulty]"
                continue

            if not in_difficulty:
                continue

            key, separator, value = line.partition(":")
            if not separator:
                continue

            key = key.strip()
            value = value.strip()

            if key == "HPDrainRate":
                hp = float(value)
            elif key == "CircleSize":
                cs = float(value)
            elif key == "OverallDifficulty":
                od = float(value)
            elif key == "ApproachRate":
                ar = float(value)

    if hp is None or cs is None or od is None:
        raise ValueError("Beatmap is missing HPDrainRate, CircleSize, or OverallDifficulty.")

    if ar is None:
        ar = od

    return BeatmapDifficulty(hp=hp, cs=cs, od=od, ar=ar)


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


def _sections(filepath):
    sections = {}
    section = None
    for raw in Path(filepath).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line
            sections.setdefault(section, [])
        elif section is not None:
            sections[section].append(line)
    return sections


def load_hit_objects(filepath):
    #Standard-mode subset: circles, L/P/B/C sliders, spinners.
    #Unknown curves, out-of-playfield heads and malformed data raise errors.
    #Lifetimes are validated by the environment using its circle timing window.
    sections = _sections(filepath)
    general = dict(tuple(part.strip() for part in line.split(":", 1)) for line in sections.get("[General]", []) if ":" in line)
    if int(general.get("Mode", "0")) != 0:
        raise ValueError("Only osu!standard Mode:0 is supported")
    difficulty = dict(tuple(part.strip() for part in line.split(":", 1)) for line in sections.get("[Difficulty]", []) if ":" in line)
    multiplier = float(difficulty.get("SliderMultiplier", "1.4"))
    if not isfinite(multiplier) or multiplier <= 0:
        raise ValueError("SliderMultiplier must be finite and positive")
    timings = []
    for line in sections.get("[TimingPoints]", []):
        fields = line.split(",")
        time, length = map(float, fields[:2])
        base = int(fields[6]) if len(fields) > 6 else 1
        if not all(map(isfinite, (time, length))) or base not in (0, 1) or (length <= 0 if base else length >= 0):
            raise ValueError("Invalid timing point")
        timings.append((time, length, base))
    if timings != sorted(timings, key=lambda point: point[0]):
        raise ValueError("Timing points must be chronological")
    objects = []
    for index, line in enumerate(sections.get("[HitObjects]", []), 1):
        try:
            fields = line.split(",")
            x, y, time = map(float, fields[:3])
            flags = int(fields[3])
            kind = flags & 139
            duration, endpoint, spans, path = 0.0, None, 1, None
            if kind == 2:
                curve = fields[5].split("|")
                controls = [(x, y)] + [tuple(map(float, point.split(":"))) for point in curve[1:]]
                spans, length = int(fields[6]), float(fields[7])
                if spans < 1:
                    raise ValueError("Invalid slider spans")
                path = SliderPath.build(curve[0], controls, length)
                endpoint = path.points[-1]
                beat_length, velocity = None, 1.0
                for timing, value, base in timings:
                    if timing > time:
                        break
                    if base:
                        beat_length, velocity = value, 1.0
                    else:
                        velocity = min(10.0, max(0.1, -100.0 / value))
                if beat_length is None:
                    raise ValueError("Slider requires a preceding uninherited timing point")
                duration = length / (multiplier * 100 * velocity) * beat_length * spans
            elif kind == 8:
                x, y = 256.0, 192.0
                duration = float(fields[5]) - time
            elif kind != 1:
                raise ValueError("Unsupported or ambiguous object type")
            if not all(map(isfinite, (x, y, time, duration))) or (kind != 1 and duration <= 0):
                raise ValueError("Object times and coordinates must be finite; duration must be positive")
            for px, py in [(x, y)]:
                if not 0 <= px <= 512 or not 0 <= py <= 384:
                    raise ValueError("Object geometry is outside the playfield")
            objects.append(HitObject(x, y, time, flags, duration, endpoint, spans, path))
        except (ValueError, IndexError) as error:
            raise ValueError(f"Hit object {index}: {error}") from error
    if objects != sorted(objects, key=lambda obj: obj.time):
        raise ValueError("Hit objects must be chronological")
    return objects


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

    parser.add_argument("--mixed", action="store_true", help="Inspect the strict mixed-object subset")
    args = parser.parse_args()

    difficulty = load_difficulty(args.osuFile)

    hit_objects = (load_hit_objects if args.mixed else load_hit_circles)(args.osuFile)
    if args.mixed:
        for obj in hit_objects:
            print(obj)
        return

    print("File:", args.osuFile)
    print(
        f"Difficulty: HP={difficulty.hp}, CS={difficulty.cs}, "
        f"OD={difficulty.od}, AR={difficulty.ar}"
    )
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
