import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from beatmapParser import HitObject, load_hit_circles


class osuAimEnv(gym.Env):
    """Circle-only aiming environment; positions are pixels and times milliseconds."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, beatmap_path=None, num_targets=10, start_on_target=False,):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

        self.beatmap_path = beatmap_path

        self.start_on_target = start_on_target

        self.width = 512
        self.height = 384

        self.cursor_speed = 1200
        self.cursor = np.zeros(2, dtype=np.float32)

        self.hit_radius = 30
        self.hit_objects = []

        self.curr_step = 0

        if num_targets < 1:
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
        self.fps = self.metadata["render_fps"]

        self.dt = 1000.0 / self.fps
        self.current_time = 0.0
        self.hit_window = 100.0

        self.hits = 0
        self.misses = 0
        self.bad_clicks = 0

        self.observation_space = spaces.Box(
            low=np.array([-512, -384, 0, -np.inf], dtype=np.float32),
            high=np.array([512, 384, 640, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

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

        self.hits = 0
        self.misses = 0
        self.bad_clicks = 0

        self.cursor = np.array(
            [
                self.np_random.uniform(0, self.width),
                self.np_random.uniform(0, self.height),
            ],
            dtype=np.float32,
        )
        if self.beatmap_path is not None:
            source_objects = load_hit_circles(self.beatmap_path)
            self.hit_objects = self._normalize_times(source_objects)
        else:
            self.hit_objects = self._generate_targets()
        if len(self.hit_objects) == 0:
            raise ValueError("Beatmap contains no hit circles.")
        self.num_targets = len(self.hit_objects)

        if self.start_on_target:
            first_object = self.hit_objects[0]
            self.cursor = np.array([
                first_object.x,
                first_object.y,
                ], dtype=np.float32)

        if options is not None and "first_target_time" in options:
            first_target_time = float(options["first_target_time"])

            if not np.isfinite(first_target_time) or first_target_time <= 0:
                raise ValueError("first target time must be finite and positive.")

            time_shift = first_target_time - self.hit_objects[0].time

            self.hit_objects = [
                    HitObject(
                        x=obj.x,
                        y=obj.y,
                        time=obj.time + time_shift,
                        object_type = obj.object_type,
                        )
                    for obj in self.hit_objects
                    ]

        observation = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        current_object = self.current_object
        if current_object is None:
            raise RuntimeError("Call reset() before stepping a new episode.")

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
        inside_timing_window = (
            abs(self.current_time - current_object.time) <= self.hit_window
        )
        clicked = action[2] > 0.5

        successful_hit = clicked and inside_circle and inside_timing_window
        expired = self.current_time > current_object.time + self.hit_window

        if clicked and not successful_hit:
            self.bad_clicks += 1

        reward = 0.0
        if successful_hit:
            self.hits += 1
            reward = 1.0
            self.current_target_i += 1
        elif expired:
            self.misses += 1
            reward = -1.0
            self.current_target_i += 1
        elif clicked:
            reward = -1.0


               

        terminated = self.current_object is None

        observation = self._get_obs()
        info = {
            "distance": float(observation[2]),
            "target_index": self.current_target_i,
            "current_time": self.current_time,
            "hits": self.hits,
            "misses": self.misses,
            "bad_clicks": self.bad_clicks,
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

        dx = current_object.x - self.cursor[0]
        dy = current_object.y - self.cursor[1]
        distance = np.hypot(dx, dy)
        time_to_target = current_object.time - self.current_time
        return np.array([dx, dy, distance, time_to_target], dtype=np.float32)

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
            target_pos = (int(current_object.x), int(current_object.y))
            # draw target
            pygame.draw.circle(
                self.window, (200, 100, 150), target_pos, self.hit_radius
            )
        # draw cursor
        pygame.draw.circle(self.window, (255, 255, 255), self.cursor.astype(int), 6)
        pygame.display.flip()
        self.clock.tick(self.fps)

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
            HitObject(
                x=obj.x,
                y=obj.y,
                time=obj.time - first_time + 1000.0,
                object_type=obj.object_type,
            )
            for obj in hit_objects
        ]
