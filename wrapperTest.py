import numpy as np

from osuEnv import osuAimEnv
from wrappers import NormalizeObservation

raw_env = osuAimEnv(num_targets=1)
scaled_env = NormalizeObservation(osuAimEnv(num_targets=1))

try:
    raw_obs, _ = raw_env.reset(seed=42)
    scaled_obs, _ = scaled_env.reset(seed=42)

    print("Original:", raw_obs)
    print("Scaled:", scaled_obs)

    np.testing.assert_allclose(
        scaled_obs,
        raw_obs / np.array([512, 384, 640, 1000], dtype=np.float32),
    )
    assert scaled_env.observation_space.contains(scaled_obs)
finally:
    raw_env.close()
    scaled_env.close()
