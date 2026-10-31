from dataclasses import replace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from beatmapParser import HitObject, load_hit_circles, load_hit_objects, load_difficulty, BeatmapDifficulty


class osuAimEnv(gym.Env):
    """osu!-style simulation. Default circles-v1 preserves the existing interface."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, beatmap_path=None, num_targets=10, start_on_target=False, bad_click_penalty=1.0, sim_hz=60, object_mode="circles"):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        if object_mode not in ("circles", "mixed"):
            raise ValueError("object_mode must be circles or mixed")
        self.object_mode = object_mode
        self.interface_version = "mixed-v3" if object_mode == "mixed" else "circles-v1"
        self.beatmap_path = beatmap_path

        self.start_on_target = start_on_target

        self.width = 512
        self.height = 384

        self.cursor_speed = 1200
        self.cursor = np.zeros(2, dtype=np.float32)

        if beatmap_path is not None:
            self.difficulty = load_difficulty(beatmap_path)
        else:
            self.difficulty = BeatmapDifficulty(hp=3.0, cs=4.2, od=6.0, ar=8.0)

        if any(not np.isfinite(value) or not 0 <= value <= 10 for value in
               (self.difficulty.hp, self.difficulty.cs, self.difficulty.od, self.difficulty.ar)):
            raise ValueError("Difficulty values must be finite and between 0 and 10")
        self.hit_radius = (54.4 - 4.48 * self.difficulty.cs) * 1.00041


        od = self.difficulty.od

        self.hit_window_300 = 80.0 - 6.0 * od
        self.hit_window_100 = 140.0 - 8.0 * od
        self.hit_window_50 = 200.0 - 10.0 * od

        self.hit_objects = []

        self.curr_step = 0

        if not isinstance(num_targets, (int, np.integer)) or num_targets < 1:
            raise ValueError("num_targets must be atleast 1.")

        self.synth_num_targets = num_targets
        self.num_targets = num_targets
        self.current_target_i = 0
        self.target_interval = 800.0

        self.max_jump_distance = 300.0
        self.jump_difficulty = 0.5

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.current_time = 0.0

        if not np.isfinite(sim_hz) or sim_hz <= 0:
            raise ValueError("sim_hz must be finite and positive.")
        
        self.sim_hz = float(sim_hz)
        self.dt = 1000.0 / self.sim_hz
        self.fps = self.metadata["render_fps"]

        self.hits = 0
        self.misses = 0
        self.bad_clicks = 0
        self.button_held = False

        if not np.isfinite(bad_click_penalty) or bad_click_penalty < 0:
            raise ValueError("bad_click_penalty must be finite and nonnegative.")
        
        self.bad_click_penalty = float(bad_click_penalty)

        self.observation_space = spaces.Box(
            low=np.array([-512, -384, 0, -np.inf, 0], dtype=np.float32),
            high=np.array([512, 384, 640, np.inf, 1], dtype=np.float32),
            dtype=np.float32,
        )

        if self.object_mode == "mixed":
            # Heads stay in-field; slider paths may leave it by at most the hit
            # radius (maximum <55 pixels across supported CS values).
            self.observation_space = spaces.Box(
                low=np.array([-567, -439, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([567, 439, np.hypot(567, 439), np.inf, 1], dtype=np.float32),
                dtype=np.float32)
            # type one-hot, duration, progress, head, coverage, signed turns,
            # cursor angle sin/cos, radius, endpoint dx/dy, span count.
            extra_low = [0, 0, 0, 0, 0, 0, 0, -np.inf, -1, -1, 0, -567, -439, 1]
            extra_high = [1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 640, 567, 439, np.inf]
            # Terminal observations are all zeros, including span count.
            extra_low[-1] = 0
            self.observation_space = spaces.Box(
                np.concatenate((self.observation_space.low, extra_low)).astype(np.float32),
                np.concatenate((self.observation_space.high, extra_high)).astype(np.float32),
                dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def current_object(self):
        if self.current_target_i >= len(self.hit_objects):
            return None
        return self.hit_objects[self.current_target_i]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_target_i = 0
        self.curr_step = 0
        self.current_time = 0.0

        self.button_held = False
        self.hits = 0
        self.misses = 0
        self.bad_clicks = 0
        self.judgement_counts = {300: 0, 100: 0, 50: 0, 0: 0}
        self.object_metrics = {kind: {"hits": 0, "misses": 0} for kind in ("circle", "slider", "spinner")}
        self._reset_object_state()

        self.cursor = np.array(
            [
                self.np_random.uniform(0, self.width),
                self.np_random.uniform(0, self.height),
            ],
            dtype=np.float32,
        )
        if self.beatmap_path is not None:
            loader = load_hit_objects if self.object_mode == "mixed" else load_hit_circles
            source_objects = loader(self.beatmap_path)
            self.hit_objects = self._normalize_times(source_objects)
        else:
            self.hit_objects = self._generate_mixed_targets() if self.object_mode == "mixed" else self._generate_targets()
        if len(self.hit_objects) == 0:
            raise ValueError("Beatmap contains no hit circles or supported objects.")
        if self.object_mode == "mixed":
            self._validate_lifetimes()
        self.num_targets = len(self.hit_objects)

        if self.start_on_target:
            first_object = self.hit_objects[0]
            self.cursor = np.array([
                first_object.x,
                first_object.y,
                ], dtype=np.float32)

        if options is not None:

            if "first_target_time" in options:
                first_target_time = float(options["first_target_time"])

                if not np.isfinite(first_target_time) or first_target_time <= 0:
                    raise ValueError("first target time must be finite and positive.")

                time_shift = first_target_time - self.hit_objects[0].time

                self.hit_objects = [
                        replace(obj, time=obj.time + time_shift)
                        for obj in self.hit_objects
                        ]

            if "start_distance" in options:
                start_distance = float(options["start_distance"])

                if not np.isfinite(start_distance) or not 0 <= start_distance <= 100:
                    raise ValueError("start_distance must be between 0 and 100")

                first_object = self.hit_objects[0]
                target_pos = np.array([
                    first_object.x, first_object.y
                    ], dtype=np.float32)

                while True:
                    angle = self.np_random.uniform(0,2*np.pi)
                    offset = start_distance * np.array([
                        np.cos(angle), np.sin(angle)
                        ], dtype=np.float32)
                    candidate = target_pos + offset

                    if (
                            0 <= candidate[0] <= self.width
                            and 0 <= candidate[1] <= self.height
                            ):
                        self.cursor = candidate.astype(np.float32)
                        break

        observation = self._get_obs()
        info = {"interface_version": self.interface_version}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        current_object = self.current_object
        if current_object is None:
            raise RuntimeError("Call reset() before stepping a new episode.")

        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must be a finite three-vector within action_space")
        previous_cursor = self.cursor.copy()
        previously_held = self.button_held
        previous_time = self.current_time
        self.curr_step += 1
        self.current_time = self.curr_step * self.dt
        movement = np.asarray(action[:2], dtype=np.float32)
        self.cursor = np.clip(
            self.cursor + movement * self.cursor_speed * (self.dt / 1000.0),
            [0, 0],
            [self.width, self.height],
        ).astype(np.float32)

        observation = self._get_obs()
        inside_circle = observation[2] <= self.hit_radius     

        timing_error = self.current_time - current_object.time
        timing_judgement = self._judge_timing(timing_error)

        button_held = bool(action[2] > 0.5)
        pressed = button_held and not self.button_held
        self.button_held = button_held

        judgement, valid_press = self._process_object(
            current_object, pressed, inside_circle, timing_judgement,
            previous_cursor, previously_held, previous_time)
        reward = 0.0
        if pressed and not valid_press:
            self.bad_clicks += 1
            reward -= self.bad_click_penalty
        judgements = []
        if judgement is not None:
            judgements.append({"index": self.current_target_i, "kind": current_object.kind, "judgement": judgement})
            success = judgement > 0
            self.hits += int(success)
            self.misses += int(not success)
            self.object_metrics[current_object.kind]["hits" if success else "misses"] += 1
            self.judgement_counts[judgement] += 1
            reward += judgement / 300.0 if success else -1.0
            self.current_target_i += 1
            self._reset_object_state()

        # A large time step can skip several lifetimes. Resolve all overdue
        # queued objects now; never award retroactive tracking or rotation.
        while self.current_object is not None:
            obj = self.current_object
            deadline = obj.end_time + (self.hit_window_50 if obj.kind == "circle" else 0)
            if self.current_time <= deadline:
                break
            judgements.append({"index": self.current_target_i, "kind": obj.kind, "judgement": 0})
            self.misses += 1
            self.object_metrics[obj.kind]["misses"] += 1
            self.judgement_counts[0] += 1
            reward -= 1
            self.current_target_i += 1
            self._reset_object_state()

        terminated = self.current_object is None

        observation = self._get_obs()
        info = {
            "distance": float(observation[2]),
            "target_index": self.current_target_i,
            "current_time": self.current_time,
            "hits": self.hits,
            "misses": self.misses,
            "bad_clicks": self.bad_clicks,
            "judgement": judgement,
            "judgements": judgements,
            "judgement_counts": self.judgement_counts.copy(),
            "object_metrics": {key: value.copy() for key, value in self.object_metrics.items()},
            "interface_version": self.interface_version
        }
        if not terminated:
            info["time_until_target"] = self.current_object.time - self.current_time
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        current_object = self.current_object
        if current_object is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        target = current_object.position_at(self.current_time)
        dx = target[0] - self.cursor[0]
        dy = target[1] - self.cursor[1]
        distance = np.hypot(dx, dy)
        time_to_target = current_object.time - self.current_time
        values = [dx, dy, distance, time_to_target, float(self.button_held)]
        if self.object_mode == "mixed":
            obj = current_object
            angle = np.arctan2(self.cursor[1] - obj.y, self.cursor[0] - obj.x)
            endpoint = obj.end_position or (obj.x, obj.y)
            values.extend([
                float(obj.kind == "circle"), float(obj.kind == "slider"), float(obj.kind == "spinner"),
                obj.duration, self._progress(obj), float(self.head_hit),
                self.tracked_ms / obj.duration if obj.duration else 0.0,
                self.rotation / (2 * np.pi), np.sin(angle), np.cos(angle),
                np.hypot(self.cursor[0] - obj.x, self.cursor[1] - obj.y),
                endpoint[0] - obj.x, endpoint[1] - obj.y, obj.spans])
        return np.array(values, dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("osu! RL env")
            self.clock = pygame.time.Clock()
        current_object = self.current_object
        if current_object is not None:
            time_until = current_object.time - self.current_time
            pygame.display.set_caption(
                f"osu! RL | Time: {self.current_time:.0f} ms "
                f"| Target in: {time_until:.0f} ms"
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        # background
        self.window.fill((20, 20, 20))
        if current_object is not None:
            target_pos = tuple(int(v) for v in current_object.position_at(self.current_time))
            if current_object.kind == "slider":
                pygame.draw.lines(self.window, (70, 90, 120), False,
                                  current_object.path.points, 8)
            elif current_object.kind == "spinner":
                pygame.draw.circle(self.window, (70, 90, 120), target_pos, 80, 3)
            # draw target
            pygame.draw.circle(
                self.window, (200, 100, 150), target_pos, self.hit_radius
            )
        # draw cursor
        pygame.draw.circle(self.window, (255, 255, 255), self.cursor.astype(int), 6)
        pygame.display.flip()
        self.clock.tick(self.sim_hz)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def _generate_targets(self):
        if self.synth_num_targets == 0:
            return []
        first_position = np.array(
            [
                self.np_random.uniform(0, self.width),
                self.np_random.uniform(0, self.height),
            ],
            dtype=np.float32,
        )
        targets = [HitObject(first_position[0], first_position[1], 1000.0, 1)]
        for i in range(1, self.synth_num_targets):
            previous = np.array([targets[i - 1].x, targets[i - 1].y])
            time_difference = self.target_interval
            available_seconds = time_difference / 1000.0
            reachable_distance = self.cursor_speed * available_seconds
            allowed_distance = min(
                self.max_jump_distance, reachable_distance * self.jump_difficulty
            )
            min_jump = min(50, allowed_distance)
            while True:
                angle = self.np_random.uniform(0, 2 * np.pi)
                jump_distance = self.np_random.uniform(min_jump, allowed_distance)
                candidate = previous + np.array(
                    [np.cos(angle) * jump_distance, np.sin(angle) * jump_distance]
                )
                x, y = candidate
                if 0 <= x <= self.width and 0 <= y <= self.height:
                    targets.append(
                        HitObject(
                            x=float(candidate[0]),
                            y=float(candidate[1]),
                            time=1000.0 + i * self.target_interval,
                            object_type=1,
                        )
                    )
                    break
        return targets

    def _normalize_times(self, hit_objects):
        first_time = hit_objects[0].time if hit_objects else 0.0
        return [
            replace(obj, time=obj.time - first_time + 1000.0)
            for obj in hit_objects
        ]

    def _reset_object_state(self):
        self.head_hit = False
        self.tracked_ms = 0.0
        self.rotation = 0.0

    def _progress(self, obj):
        return float(np.clip((self.current_time - obj.time) / obj.duration, 0, 1)) if obj.duration else 0.0

    def _validate_lifetimes(self):
        for index, obj in enumerate(self.hit_objects, 1):
            if obj.path is not None:
                points = np.asarray(obj.path.points)
                nearest = np.clip(points, [0, 0], [self.width, self.height])
                if np.any(np.linalg.norm(points - nearest, axis=1) > self.hit_radius):
                    raise ValueError(f"Hit object {index}: slider path is outside cursor reach")
        for previous, current in zip(self.hit_objects, self.hit_objects[1:]):
            deadline = previous.end_time + (self.hit_window_50 if previous.kind == "circle" else 0)
            if current.time <= deadline:
                raise ValueError("Overlapping object lifetimes are unsupported in mixed mode")

    def _generate_mixed_targets(self):
        objects = []
        time = 1000.0
        for i in range(self.synth_num_targets):
            x, y = self.np_random.uniform([100, 100], [350, 280])
            if i % 3 == 0:
                obj = HitObject(float(x), float(y), time, 1)
            elif i % 3 == 1:
                obj = HitObject(float(x), float(y), time, 2, 1600.0, (float(x + 100), float(y)), 2)
            else:
                obj = HitObject(256, 192, time, 8, 2000.0)
            objects.append(obj)
            time = obj.end_time + self.target_interval
        return objects

    def _process_object(self, obj, pressed, inside, timing, old_cursor, old_held, old_time):
        if obj.kind == "circle":
            valid = pressed and inside and timing is not None
            return (timing if valid else 0 if self.current_time - obj.time > self.hit_window_50 else None), valid
        overlap = max(0.0, min(self.current_time, obj.end_time) - max(old_time, obj.time))
        if obj.kind == "slider":
            head_distance = np.hypot(self.cursor[0] - obj.x, self.cursor[1] - obj.y)
            valid = pressed and not self.head_hit and head_distance <= self.hit_radius and timing is not None and self.current_time <= obj.end_time
            self.head_hit = self.head_hit or valid
            if self.head_hit and self.button_held and inside:
                self.tracked_ms += overlap
            if self.current_time >= obj.end_time:
                return (300 if self.head_hit and self.tracked_ms / obj.duration >= 0.9 - 1e-9 else 0), valid
            return None, valid
        # Spinner: signed shortest angular deltas. Reversals cancel, avoiding
        # oscillation farming. Both samples must be outside the 30-pixel deadzone.
        before = old_cursor - np.array([obj.x, obj.y])
        after = self.cursor - np.array([obj.x, obj.y])
        segment = after - before
        length_squared = float(segment @ segment)
        fraction = float(np.clip(-(before @ segment) / length_squared, 0, 1)) if length_squared else 0
        closest_radius = np.linalg.norm(before + fraction * segment)
        if overlap and old_held and self.button_held and closest_radius >= 30:
            delta = np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0])
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            self.rotation += delta * overlap / self.dt
        valid = pressed and obj.time <= self.current_time <= obj.end_time
        if self.current_time >= obj.end_time:
            required = obj.duration / 1000.0 * 2 * np.pi  # one turn/second
            return (300 if abs(self.rotation) >= required - 1e-6 else 0), valid
        return None, valid

    def _judge_timing(self, timing_error):
        error = abs(timing_error)

        if error <= self.hit_window_300:
            return 300
        elif error <= self.hit_window_100:
            return 100
        elif error <= self.hit_window_50:
            return 50

        return None
