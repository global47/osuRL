from osuEnv import osuAimEnv
import numpy as np

env = osuAimEnv(
    render_mode="human",
    beatmap_path="FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu",
)
termined = False


obs, info = env.reset(seed=42)

print("Starting obsverations:")
print(obs)

while True:
    dx, dy, distance, time_until = obs

    move_x = 0.0
    move_y = 0.0

    if distance > 0 and time_until > 0:
        time_until_s = time_until / 1000

        required_speed = distance / time_until_s

        speed_frac = required_speed / env.cursor_speed

        speed_frac = np.clip(speed_frac, 0.0, 1.0)

        move_x = (dx / distance) * speed_frac
        move_y = (dy / distance) * speed_frac

    else:
        move_x = dx / distance
        move_y = dy / distance

    click = 0.0

    if distance <= env.hit_radius and abs(time_until) <= env.hit_window:
        click = 1.0

    action = np.array([move_x, move_y, click], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

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
