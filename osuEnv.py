import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy._core.numeric import float32
import pygame

class osuAimEnv(gym.Env):
    
    def __init__(self, render_mode=None):
        super().__init__()

        self.width = 512
        self.height = 384
        self.cursor_speed = 20
        self.hit_radius = 30

        self.max_steps = 200
        self.curr_step = 0

        self.cursor = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

        self.num_targets = 10
        self.current_target_i = 0
        self.targets = np.empty((0,2), dtype=np.float32)
        

        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.observation_space = spaces.Box(
                low=np.array(
                    [-512,-384,0],
                    dtype=np.float32),
                high=np.array(
                    [512,384,640],
                    dtype=np.float32
                    ),
                dtype=np.float32
                )

        self.action_space = spaces.Box(
                low=np.array(
                    [-1,-1, 0],
                    dtype=np.float32),
                high=np.array(
                    [1,1,1],
                    dtype=np.float32
                    ),
                dtype=np.float32
                )

       

    def reset(self,
              *,
              seed: int | None = None,
              options:dict | None = None):

        super().reset(seed=seed)

        self.current_target_i = 0
        self.curr_step = 0

        self.cursor = np.array([
             self.np_random.uniform(0, self.width),
             self.np_random.uniform(0, self.height)
             ], dtype=np.float32)
   
        self.targets = np.column_stack((
            self.np_random.uniform(
                0,
                self.width,
                size = self.num_targets
                ),
            self.np_random.uniform(
                0,
                self.height,
                size = self.num_targets
                )
            )).astype(np.float32)


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

        dx = action[0]
        dy = action[1]

        clicked = action[2] > 0.5

        # move
        self.cursor[0] += dx * self.cursor_speed
        self.cursor[1] += dy * self.cursor_speed

        # clip
        self.cursor[0] = np.clip(
                self.cursor[0],
                0,
                self.width
                )

        self.cursor[1] = np.clip(
                self.cursor[1],
                0,
                self.height
                )

        distance = np.linalg.norm(
                self.cursor - self.target
                ) 

        inside_circle = distance <= self.hit_radius

        # rewards!
        reward = 0.0
        terminated = False

        if clicked:
        
            if inside_circle:
                reward = 1.0

                self.current_target_i += 1

                if self.current_target_i >= self.num_targets:
                    terminated = True
                else:
                    self.target = self.targets[
                        self.current_target_i
                        ]
        else:
            reward = -1.0

        truncated = (
                self.curr_step >= self.max_steps
                )

        observation = self._get_obs()

        if not terminated:
            
            new_distance = np.linalg.norm(
                    self.cursor - self.target
                    ) if not terminated else 0.0
        else:
            new_distance = 0.0


        info = {
                "distance": new_distance,
                "target_index": self.current_target_i
               }

        if self.render_mode == "human":
            self.render()

        return (
                observation,
                reward,
                terminated,
                truncated,
                info
                )



    def _get_obs(self):
        dx = self.target[0] - self.cursor[0]
        dy = self.target[1] - self.cursor[1]
        distance = np.hypot(dx,dy)
        return np.array([
            dx,
            dy,
            distance
            ], dtype=np.float32)


    def render(self):

        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                    (self.width, self.height)
                    )

            pygame.display.set_caption("osu! RL env")

            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # background
        self.window.fill((20, 20, 20))
        
        # draw target
        pygame.draw.circle(
                self.window,
                (200, 100, 150),
                self.target.astype(int),
                self.hit_radius
                )

        # draw cursor
        pygame.draw.circle(
                self.window,
                (255,255,255),
                self.cursor.astype(int),
                6
                )

        pygame.display.flip()

        self.clock.tick(30)

    def close(self):
        
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
