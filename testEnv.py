from osuEnv import osuAimEnv

env = osuAimEnv()


obs, info = env.reset(seed=42)

print("Starting obsverations:")
print(obs)


for step in range(10):
    
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(
            action
            )

    print()
    print("Step: ", step)
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Distance:", info["distance"])

    if terminated or truncated:
        print("Episode finished.")
        break
