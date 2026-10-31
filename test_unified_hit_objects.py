from pathlib import Path

import numpy as np
import pytest

from beatmapParser import HitObject, load_hit_circles
from osuEnv import osuAimEnv


ROOT = Path(__file__).parent
BEATMAP = ROOT / "FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu"


def test_synthetic_reset_uses_hit_objects_only():
    env = osuAimEnv()

    observation, _ = env.reset(seed=42)

    assert observation.shape == (5,)
    assert len(env.hit_objects) == 10
    assert all(isinstance(obj, HitObject) for obj in env.hit_objects)
    assert [obj.time for obj in env.hit_objects] == [1000.0 + i * 800.0 for i in range(10)]
    assert not hasattr(env, "target")
    assert not hasattr(env, "targets")
    assert not hasattr(env, "target_times")
    assert env.current_object is env.hit_objects[0]


def test_synthetic_generation_is_seeded():
    first = osuAimEnv()
    second = osuAimEnv()

    first.reset(seed=42)
    second.reset(seed=42)

    first_objects = [(obj.x, obj.y, obj.time) for obj in first.hit_objects]
    second_objects = [(obj.x, obj.y, obj.time) for obj in second.hit_objects]

    assert first_objects == second_objects


def test_parsed_reset_normalizes_without_mutating_parser_result():
    source_objects = load_hit_circles(BEATMAP)
    first_source_time = source_objects[0].time
    env = osuAimEnv(beatmap_path=str(BEATMAP))

    observation, _ = env.reset(seed=42)

    assert observation.shape == (5,)
    assert len(env.hit_objects) == len(source_objects)
    assert env.hit_objects[0].time == 1000.0
    assert source_objects[0].time == first_source_time
    assert all(isinstance(obj, HitObject) for obj in env.hit_objects)
    assert env.current_object is env.hit_objects[0]


def test_timeout_advances_the_canonical_object_list():
    env = osuAimEnv()
    env.reset(seed=42)

    deadline = env.current_object.time + env.hit_window_50
    steps_until_expired = int(np.floor(deadline / env.dt)) + 1
    for _ in range(steps_until_expired):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )

    assert not terminated
    assert not truncated
    assert info["target_index"] == 1
    assert env.current_object is env.hit_objects[1]
    assert observation.shape == (5,)


def test_final_hit_returns_terminal_observation_and_requires_reset():
    import pytest

    env = osuAimEnv()
    env.reset(seed=42)
    env.hit_objects = [HitObject(100, 100, env.dt, 1)]
    env.cursor[:] = 100

    observation, reward, terminated, truncated, info = env.step(
        np.array([0, 0, 1], dtype=np.float32)
    )

    assert reward == 1.0
    assert terminated and not truncated
    assert env.current_object is None
    assert np.array_equal(observation, np.zeros(5, dtype=np.float32))
    assert info["distance"] == 0.0
    assert "time_until_target" not in info
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3, dtype=np.float32))


def test_final_timeout_terminates():
    env = osuAimEnv()
    env.reset(seed=42)
    env.hit_objects = [HitObject(100, 100, -200, 1)]

    observation, reward, terminated, _, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert terminated
    assert reward == -1.0
    assert env.observation_space.contains(observation)
    assert isinstance(info["distance"], float)


def test_long_break_is_inside_observation_space():
    env = osuAimEnv()
    env.reset(seed=42)
    env.hit_objects = [HitObject(100, 100, 30000, 1)]
    assert env.observation_space.contains(env._get_obs())


def test_parser_ignores_comments_and_following_sections(tmp_path):
    beatmap = tmp_path / "sample.osu"
    beatmap.write_text(
        "[HitObjects]\n// Comment\n100,200,1000,5,0\n"
        "200,100,2000,2,0\n[Other]\nnot an object\n",
        encoding="utf-8-sig",
    )
    assert load_hit_circles(beatmap) == [HitObject(100, 200, 1000, 5)]


def test_empty_beatmap_has_clear_error(tmp_path):
    import pytest

    beatmap = tmp_path / "empty.osu"
    beatmap.write_text(
        "[Difficulty]\nHPDrainRate:5\nCircleSize:5\n"
        "OverallDifficulty:5\n[HitObjects]\n"
    )
    with pytest.raises(ValueError, match="no hit circles"):
        osuAimEnv(beatmap_path=beatmap).reset()


def test_demo_action_at_target_is_finite():
    from testEnv import aim_action

    env = osuAimEnv()
    action = aim_action(np.zeros(5, dtype=np.float32), env)
    assert env.action_space.contains(action)
    assert np.array_equal(action, [0, 0, 1])


def test_gymnasium_contract():
    from gymnasium.utils.env_checker import check_env

    check_env(osuAimEnv(), skip_render_check=True)


@pytest.mark.parametrize(
    "error, button, on_target, expected_judgement, expected_reward",
    [
        (-44.0, 1, True, 300, 1.0),
        (44.0, 1, True, 300, 1.0),
        (44.01, 1, True, 100, 1 / 3),
        (92.0, 1, True, 100, 1 / 3),
        (92.01, 1, True, 50, 1 / 6),
        (140.0, 1, True, 50, 1 / 6),
        (140.01, 0, True, 0, -1.0),
        (-140.01, 1, True, None, -1.0),
        (0.0, 0, True, None, 0.0),
        (0.0, 1, False, None, -1.0),
    ],
)
def test_step_timing_judgements(
    error, button, on_target, expected_judgement, expected_reward
):
    env = osuAimEnv(num_targets=1)
    try:
        env.reset(seed=42)
        # Evaluate at 1000 ms with a controlled signed timing error.
        env.curr_step = 59
        env.hit_objects = [HitObject(100, 100, 60 * env.dt - error, 1)]
        env.cursor[:] = 100 if on_target else 300
        _, reward, terminated, _, info = env.step(
            np.array([0, 0, button], dtype=np.float32)
        )
        assert info["judgement"] == expected_judgement
        assert reward == pytest.approx(expected_reward)
        assert terminated == (expected_judgement is not None)
        assert info["hits"] == int(expected_judgement in (300, 100, 50))
        assert info["misses"] == int(expected_judgement == 0)
    finally:
        env.close()


def test_second_circle_requires_release_and_repress():
    env = osuAimEnv(num_targets=2)
    try:
        env.reset(seed=42)
        env.hit_objects = [
            HitObject(100, 100, env.dt, 1),
            HitObject(100, 100, 4 * env.dt, 1),
        ]
        env.cursor[:] = 100

        for button, expected_hits in [(1, 1), (1, 1), (0, 1), (1, 2)]:
            action = np.array([0, 0, button], dtype=np.float32)
            _, _, terminated, _, info = env.step(action)

            assert info["hits"] == expected_hits
            assert terminated == (expected_hits == 2)
    finally:
        env.close()
