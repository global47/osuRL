import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_env = env.unwrapped

        self.scale = np.array(
            [base_env.width, base_env.height, np.hypot(base_env.width, base_env.height), 1000.0, 1.0],
            dtype=np.float32,
        )

        if base_env.object_mode == "mixed":
            self.scale = np.concatenate((self.scale, np.array(
                [1, 1, 1, 1000, 1, 1, 1, 1, 1, 1, 640, 512, 384, 1], dtype=np.float32)))

        self.observation_space = spaces.Box(
            low=env.observation_space.low / self.scale,
            high=env.observation_space.high / self.scale,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32) / self.scale


class TimingOnlyAction(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def action(self, action):
        click = float(action[0])
        return np.array([0.0, 0.0, click], dtype=np.float32)

class NearbyStart(gym.Wrapper):
    def __init__(self, env, distance=45.0):
        super().__init__(env)
        self.distance = distance

    def reset(self, *, seed=None, options=None):
        reset_options = {"start_distance": self.distance}

        if options is not None:
            reset_options.update(options)

        return self.env.reset(seed=seed, options=reset_options)
