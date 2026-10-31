import numpy as np

from osuEnv import osuAimEnv
from wrappers import NormalizeObservation

env = NormalizeObservation(
    osuAimEnv(num_targets=1, start_on_target=True)
)

try:
    observation, _ = env.reset(seed=42)
    print("After reset:", observation[4])

    for button in [1, 1, 0, 1]:
        action = np.array([0, 0, button], dtype=np.float32)
        observation, reward, _, _, info = env.step(action)

        assert env.observation_space.contains(observation)
        assert observation[4] == float(button)

        print(
            f"action={button}, "
            f"observed_hold={observation[4]}, "
            f"bad_clicks={info['bad_clicks']}"
        )
finally:
    env.close()