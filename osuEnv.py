import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy._core.numeric import float32

class osuAimEnv(gym.Env):
    
    def __init__(self):
        super().__init__()

        self.width = 512
        self.height = 384
        self.cursor_speed = 20
        self.hit_radius = 30

        self.max_steps = 200
        self.curr_step = 0

        self.cursor = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

        self.observation_space = spaces.Box(
                low=np.array(
                    [0,0,0,0],
                    dtype=np.float32),
                high=np.array(
                    [512,384,512,384],
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

        self.cursor = np.array([
             self.np_random.uniform(0, self.width),
             self.np_random.uniform(0, self.height)
             ], dtype=np.float32)
   
        self.target = np.array([
            self.np_random.uniform(0,self.width),
            self.np_random.uniform(0,self.height)
            ], dtype=np.float32)

        observation = self._get_obs()

        info = {}

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

        if clicked:

            if inside_circle:
                reward = 1.0
            else:
                reward = -1.0

        terminated = clicked and inside_circle

        truncated = (
                self.curr_step >= self.max_steps
                )

        observation = self._get_obs()

        info = {
                "distance": distance
                }

        return (
                observation,
                reward,
                terminated,
                truncated,
                info
                )




    def _get_obs(self):
        return np.array([
            self.cursor[0],
            self.cursor[1],
            self.target[0],
            self.target[1]
            ], dtype=np.float32)


