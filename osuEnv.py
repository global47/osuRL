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

        

        self.cursor = np.zeros(2, dtype=np.float32)
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
        self.target = np.zeros(2, dtype=np.float32)

        self.observation_space = spaces.Box(
                low=np.array([0,0,0,0], dtype=np.float32),
                high=np.array([512,384,512,384], dtype=np.float32),
                dtype=np.float32
                )

        self.action_space = spaces.Box(
                low=np.array([-1,-1, 0], dtype=np.float32),
                high=np.array([1,1,1], dtype=np.float32),
                dtype=float32
                )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor = np.array([
             self.np_random.uniform(0, self.width),
             self.np_random.uniform(512, self.height)
             ], dtype=np.float32)
   
        self.target = np.array([
            self.np_random.uniform(0,self.width),
            self.np_random.uniform(0,self.height)
            ], dtype=np,float32)

        observation = self._get_obs()

        info = {}

        return observation, info


    def step(self, action):
        pass

    def _get_obs(self):
        return np.array([
            self.cursor[0],
            self.cursor[1],
            self.target[0],
            self.target[1]
            ], dtype=np.float32)


