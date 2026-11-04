# this is a scripted policy that will always perform the optimal action.
import numpy as np


def scripted_action(observation, env):
    base = env.unwrapped
    obj = base.current_object
    next_time = base.current_time + base.dt
    target = np.array(obj.position_at(next_time))
    if obj.kind == "spinner":
        angle = np.arctan2(base.cursor[1] - obj.y, base.cursor[0] - obj.x)
        if next_time >= obj.time:
            angle += 2 * np.pi * 1.5 * base.dt / 1000
        target = np.array([obj.x, obj.y]) + 80 * np.array([np.cos(angle), np.sin(angle)])
        click = float(next_time >= obj.time)
    elif obj.kind == "slider" and base.head_hit:
        click = 1.0
    else:
        target = np.array([obj.x, obj.y])
        click = float(next_time >= obj.time and not base.button_held)
    movement = np.clip((target - base.cursor) / (base.cursor_speed * base.dt / 1000), -1, 1)
    if obj.kind != "spinner" and not base.head_hit:
        predicted = base.cursor + movement * base.cursor_speed * base.dt / 1000
        click *= float(np.linalg.norm(predicted - target) <= base.hit_radius)
    return np.array([*movement, click], dtype=np.float32)
