from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from beatmapParser import HitObject, load_hit_objects
from evaluate import evaluate
from osuEnv import osuAimEnv
from policies import scripted_action
from wrappers import NormalizeObservation

FIXTURE = Path(__file__).parent / "fixtures/mixed.osu"


@pytest.mark.parametrize("hz", [60, 120, 240])
@pytest.mark.parametrize("beatmap", [None, FIXTURE])
def test_scripted_success_at_supported_rates(hz, beatmap):
    env = osuAimEnv(object_mode="mixed", num_targets=9, sim_hz=hz, beatmap_path=beatmap)
    def checked_policy(obs, env):
        assert env.observation_space.contains(obs)
        return scripted_action(obs, env)
    result = evaluate(checked_policy, range(3), env)
    assert result["hit_rate"] == result["accuracy"] == 1
    assert result["bad_clicks"] == 0
    assert all(metrics["hit_rate"] == 1 for metrics in result["by_type"].values())
    assert result["targets"] == (12 if beatmap else 27)


def test_all_objects_miss_without_input():
    result = evaluate(lambda obs, env: np.zeros(3, dtype=np.float32), [0],
                      osuAimEnv(object_mode="mixed", num_targets=3))
    assert result["misses"] == 3
    assert result["hits"] == result["bad_clicks"] == 0
    assert result["mean_reward"] == -3


def test_mixed_reset_and_wrapped_contract():
    env = NormalizeObservation(osuAimEnv(object_mode="mixed", num_targets=3))
    check_env(env.unwrapped, skip_render_check=True)
    first, _ = env.reset(seed=12)
    original = list(env.unwrapped.hit_objects)
    for _ in range(100):
        env.step(np.array([0, 0, 1], dtype=np.float32))
    second, _ = env.reset(seed=12)
    assert np.array_equal(first, second)
    assert original == env.unwrapped.hit_objects
    assert env.observation_space.contains(second)
    assert env.unwrapped.rotation == env.unwrapped.tracked_ms == 0
    assert not env.unwrapped.head_hit
    assert second.shape == (19,)
    assert sum(env.unwrapped.judgement_counts.values()) == 0


def test_slider_repeat_positions():
    obj = HitObject(100, 100, 1000, 2, 1000, (200, 100), 2)
    assert [obj.position_at(t) for t in [0, 1000, 1250, 1500, 1750, 2000, 3000]] == [
        (100, 100), (100, 100), (150, 100), (200, 100), (150, 100), (100, 100), (100, 100)]
    assert replace(obj, spans=3).position_at(2000) == (200, 100)


@pytest.mark.parametrize("failure", ["release", "off_path", "no_head"])
def test_slider_failure_modes(failure):
    env = osuAimEnv(object_mode="mixed", num_targets=1)
    env.reset(seed=0)
    env.hit_objects = [HitObject(100, 100, 500, 2, 1000, (200, 100))]
    while True:
        action = scripted_action(None, env)
        if failure == "no_head":
            action[2] = 0
        elif env.current_time >= 750:
            if failure == "release":
                action[2] = 0
            else:
                action[:2] = [0, 1]
        obs, _, done, _, info = env.step(action)
        if done:
            break
    assert info["hits"] == 0 and info["misses"] == 1
    assert env.observation_space.contains(obs)
    assert np.all(obs == 0)


def test_slider_needs_fresh_head_press():
    env = osuAimEnv(object_mode="mixed", num_targets=1)
    env.reset(seed=0)
    env.hit_objects = [HitObject(100, 100, 1000, 2, 1000, (200, 100))]
    env.cursor[:] = 100
    for _ in range(70):
        env.step(np.array([0, 0, 1], dtype=np.float32))
    assert not env.head_hit
    assert env.bad_clicks == 1


@pytest.mark.parametrize("direction", [-1, 1])
def test_spinner_wraparound_and_direction(direction):
    env = osuAimEnv(object_mode="mixed", num_targets=1, sim_hz=120)
    env.reset(seed=0)
    env.hit_objects = [HitObject(256, 192, 0, 8, 1000)]
    env.cursor[:] = [176, 192]
    env.button_held = True
    for i in range(120):
        angle = np.pi + direction * (i + 1) * 2 * np.pi * 1.5 / 120
        target = np.array([256, 192]) + 80 * np.array([np.cos(angle), np.sin(angle)])
        action = np.array([*((target - env.cursor) / (env.cursor_speed * env.dt / 1000)), 1], dtype=np.float32)
        _, _, done, _, info = env.step(action)
    assert done and info["hits"] == 1


def test_spinner_center_does_not_gain_rotation():
    env = osuAimEnv(object_mode="mixed", num_targets=1)
    env.reset(seed=0)
    env.hit_objects = [HitObject(256, 192, 0, 8, 1000)]
    env.cursor[:] = [256, 192]
    for i in range(60):
        _, _, done, _, info = env.step(np.array([0, 0, 1], dtype=np.float32))
    assert done and info["misses"] == 1


def test_parser_timing_velocity_repeat_and_redline_reset(tmp_path):
    objects = load_hit_objects(FIXTURE)
    assert [obj.kind for obj in objects] == ["circle", "slider", "spinner", "circle"]
    assert objects[1].duration == 500  # 140/(1.4*100*2)*500*2
    assert objects[2].duration == 2000
    source = FIXTURE.read_text().replace("350,250,7000,1,0", "100,100,7000,2,0,L|240:100,1,140")
    path = tmp_path / "reset.osu"
    path.write_text(source)
    assert load_hit_objects(path)[-1].duration == 500


@pytest.mark.parametrize("old,new,message", [
    ("L|290:150", "X|290:150", "Unsupported slider"),
    ("Mode:0", "Mode:3", "Mode:0"),
    ("2,140", "0,140", "spans"),
    ("2,140", "2,nan", "geometry"),
    ("0,6000", "0,3000", "duration"),
])
def test_unsupported_or_malformed_maps_fail_clearly(tmp_path, old, new, message):
    path = tmp_path / "bad.osu"
    path.write_text(FIXTURE.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_hit_objects(path)


def test_overlap_is_rejected(tmp_path):
    path = tmp_path / "overlap.osu"
    path.write_text(FIXTURE.read_text().replace("350,250,7000", "350,250,5000"))
    with pytest.raises(ValueError, match="Overlapping"):
        osuAimEnv(object_mode="mixed", beatmap_path=path).reset()


def test_time_shift_preserves_duration_endpoint_and_spans():
    env = osuAimEnv(object_mode="mixed", beatmap_path=FIXTURE)
    env.reset(seed=0, options={"first_target_time": 1500})
    source = load_hit_objects(FIXTURE)
    assert env.hit_objects == [replace(obj, time=obj.time + 500) for obj in source]


@pytest.mark.parametrize("action", [[0, 0], [0, 0, np.nan], [2, 0, 0]])
def test_invalid_action_has_no_side_effects(action):
    env = osuAimEnv(object_mode="mixed")
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="Action"):
        env.step(action)
    assert env.current_time == 0
    assert np.array_equal(obs, env._get_obs())


def test_large_step_resolves_all_elapsed_objects():
    env = osuAimEnv(object_mode="mixed", num_targets=3, sim_hz=0.1)
    env.reset(seed=0)
    obs, reward, done, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert done and info["misses"] == 3
    assert reward == -3
    assert len(info["judgements"]) == 3
    assert env.observation_space.contains(obs)


def test_spinner_center_crossing_does_not_count_as_half_turn():
    env = osuAimEnv(object_mode="mixed", num_targets=1, sim_hz=10)
    env.reset(seed=0)
    env.hit_objects = [HitObject(256, 192, 0, 8, 1000)]
    env.cursor[:] = [206, 192]
    env.button_held = True
    env.step(np.array([100 / 120, 0, 1], dtype=np.float32))
    assert env.rotation == 0


def test_normalized_mixed_episode_stays_in_space():
    env = NormalizeObservation(osuAimEnv(object_mode="mixed", num_targets=3))
    obs, _ = env.reset(seed=0)
    while True:
        assert env.observation_space.contains(obs)
        obs, _, done, _, _ = env.step(scripted_action(obs, env))
        if done:
            assert env.observation_space.contains(obs)
            break


@pytest.mark.parametrize("tracked_steps,expected", [(90, 1), (89, 0)])
def test_slider_ninety_percent_boundary(tracked_steps, expected):
    env = osuAimEnv(object_mode="mixed", num_targets=1, sim_hz=100)
    env.reset(seed=0)
    env.hit_objects = [HitObject(100, 100, 100, 2, 1000, (200, 100))]
    env.cursor[:] = 100
    for step in range(110):
        action = scripted_action(None, env)
        if step >= 10 + tracked_steps:
            action[2] = 0
        _, _, done, _, info = env.step(action)
    assert done and info["hits"] == expected
