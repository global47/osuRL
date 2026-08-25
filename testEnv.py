from osuEnv import osuAimEnv
import numpy as np

env = osuAimEnv(render_mode = "human")
termined = False


obs, info = env.reset(seed=41)

print("Starting obsverations:")
print(obs)

while(True):

    dx, dy, distance, time_until = obs

    max_movement = (
            env.cursor_speed * env.dt / 1000
            )

    if distance > 0:
        move_x = dx / distance
        move_y = dy / distance
    else:
        move_x = dx / max_movement
        move_y = dy / max_movement



    click = 0.0

    if (
            distance <= env.hit_radius
            and abs(time_until) <= env.hit_window
    ):
        click = 1.0

    action = np.array([
        move_x,
        move_y,
        click
        ], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if terminated or truncated:
        break


    print(
        "Step:",
        env.curr_step,
        "Reward:",
        reward,
        "Distance:",
        info["distance"],
        "terminated",
        terminated,
        "truncated",
        truncated,
        "target:",
        info["target_index"],
    )


env.close()
