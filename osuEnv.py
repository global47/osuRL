import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy._core.numeric import float32
import pygame
from beatmapParser import load_hit_circles


class osuAimEnv(gym.Env):
    def __init__(self, render_mode=None, beatmap_path=None):
        super().__init__()

        self.beatmap_path = beatmap_path

        self.width = 512
        self.height = 384
        self.cursor_speed = 1200
        self.hit_radius = 30

        self.curr_step = 0

        self.cursor = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

        self.num_targets = 10
        self.current_target_i = 0

        self.targets = np.empty((0, 2), dtype=np.float32)
        self.target_times = np.empty(0, dtype=np.float32)
        self.target_interval = 800.0

        self.max_jump_distance = 300.0
        self.jump_difficulty = 0.5

        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.fps = 60
        self.dt = 1000.0 / self.fps

        self.current_time = 0.0

        self.hit_window = 100.0

        self.observation_space = spaces.Box(
            low=np.array([-512, -384, 0, -10000], dtype=np.float32),
            high=np.array([512, 384, 640, 10000], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):

        super().reset(seed=seed)

        self.current_target_i = 0
        self.curr_step = 0
        self.current_time = 0.0

        self.cursor = np.array(
            [
                self.np_random.uniform(0, self.width),
                self.np_random.uniform(0, self.height),
            ],
            dtype=np.float32,
        )

        if self.beatmap_path is not None:
            self.targets, self.target_times = load_hit_circles(self.beatmap_path)
            self.num_targets = len(self.targets)

            first_time = self.target_times[0]

            self.target_times = self.target_times - first_time + 1000.0

        else:
            self.target_times = np.arange(
                1000,
                1000 + self.num_targets * self.target_interval,
                self.target_interval,
                dtype=np.float32,
            )

            self.targets = self._generate_targets()

        self.target = self.targets[self.current_target_i]

        observation = self._get_obs()

        print("Targets:")
        print(self.targets)

        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):

        self.curr_step += 1

        self.current_time += self.dt

        target_time = self.target_times[self.current_target_i]

        timing_err = abs(self.current_time - target_time)

        inside_timing_window = timing_err <= self.hit_window

        dt_seconds = self.dt / 1000.0

        dx = action[0]
        dy = action[1]

        clicked = action[2] > 0.5

        # move

        self.cursor[0] += dx * self.cursor_speed * dt_seconds

        self.cursor[1] += dy * self.cursor_speed * dt_seconds

        # clip
        self.cursor[0] = np.clip(self.cursor[0], 0, self.width)

        self.cursor[1] = np.clip(self.cursor[1], 0, self.height)

        distance = np.linalg.norm(self.cursor - self.target)

        inside_circle = distance <= self.hit_radius

        # rewards!
        reward = 0.0
        terminated = False

        if clicked:
            if inside_circle and inside_timing_window:
                reward = 1.0

                self.current_target_i += 1

                if self.current_target_i >= self.num_targets:
                    terminated = True
                else:
                    self.target = self.targets[self.current_target_i]
            else:
                reward = -1.0

        truncated = self.current_target_i >= self.num_targets

        observation = self._get_obs()

        if (
            not terminated
            and self.current_time
            > self.target_times[self.current_target_i] + self.hit_window
        ):
            reward = -1.0

            self.current_target_i += 1

            if self.current_target_i >= self.num_targets:
                terminated = True
            else:
                self.target = self.targets[self.current_target_i]

        new_distance = (
            np.linalg.norm(self.cursor - self.target) if not terminated else 0.0
        )

        info = {
            "distance": new_distance,
            "target_index": self.current_target_i,
            "current_time": self.current_time,
        }

        if not terminated:
            info["time_until_target"] = (
                self.target_times[self.current_target_i] - self.current_time
            )

        if self.render_mode == "human":
            self.render()

        return (observation, reward, terminated, truncated, info)

    def _get_obs(self):
        dx = self.target[0] - self.cursor[0]
        dy = self.target[1] - self.cursor[1]

        distance = np.hypot(dx, dy)

        if self.current_target_i < self.num_targets:
            time_to_target = (
                self.target_times[self.current_target_i] - self.current_time
            )
        else:
            time_to_target = 0.0

        return np.array([dx, dy, distance, time_to_target], dtype=np.float32)

    def render(self):

        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))

            pygame.display.set_caption("osu! RL env")

            self.clock = pygame.time.Clock()

        if self.current_target_i < self.num_targets:
            time_until = self.target_times[self.current_target_i] - self.current_time

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

        # draw target
        pygame.draw.circle(
            self.window, (200, 100, 150), self.target.astype(int), self.hit_radius
        )

        # draw cursor
        pygame.draw.circle(self.window, (255, 255, 255), self.cursor.astype(int), 6)

        pygame.display.flip()

        self.clock.tick(30)

    def close(self):

        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def _generate_targets(self):

        targets = np.zeros((self.num_targets, 2), dtype=np.float32)

        targets[0] = np.array(
            [
                self.np_random.uniform(0, self.width),
                self.np_random.uniform(0, self.height),
            ],
            dtype=np.float32,
        )

        for i in range(1, self.num_targets):
            previous = targets[i - 1]

            time_difference = self.target_times[i] - self.target_times[i - 1]

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
                    targets[i] = candidate
                    break

        return targets
