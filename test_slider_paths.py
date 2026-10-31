from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

from beatmapParser import HitObject, load_hit_objects
from evaluate import evaluate
from osuEnv import osuAimEnv
from policies import scripted_action
from sliderPath import SliderPath

ROOT = Path(__file__).parent
MAP = ROOT / 'FUNKIST - Snow fairy (TV Size) (Yumerios) [Easy].osu'
CURVES = ROOT / 'fixtures/curves.osu'


def test_linear_corners_trim_extend_and_constant_distance():
    path = SliderPath.build('L', [(0, 0), (30, 0), (30, 40)], 100)
    assert path.position_at(.3) == (30, 0)
    assert path.position_at(.7) == (30, 40)
    assert path.position_at(1) == (30, 70)
    assert path.position_at(-1) == (0, 0)
    assert path.position_at(2) == (30, 70)
    short = SliderPath.build('L', [(0, 0), (30, 0), (30, 40)], 50)
    assert short.points[-1] == (30, 20)
    assert short.distances[-1] == 50


@pytest.mark.parametrize('middle', [(0, 100), (0, -100)])
def test_circular_arc_direction_radius_and_equal_distance(middle):
    path = SliderPath.build('P', [(100, 0), middle, (-100, 0)], 100 * pi)
    samples = np.array([path.position_at(t) for t in np.linspace(0, 1, 101)])
    assert np.max(np.abs(np.linalg.norm(samples, axis=1) - 100)) < .06
    assert samples[50] == pytest.approx(middle, abs=.06)
    distances = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    assert np.ptp(distances) < .01
    assert samples[-1] == pytest.approx((-100, 0), abs=.06)


def test_arc_takes_major_sweep_through_middle_control():
    path = SliderPath.build('P', [(100, 0), (-100, 0), (0, 100)], 150 * pi)
    assert path.position_at(2 / 3) == pytest.approx((-100, 0), abs=.08)
    assert path.position_at(1 / 3) == pytest.approx((0, -100), abs=.08)
    assert path.position_at(1) == pytest.approx((0, 100), abs=.08)


def test_bezier_red_anchor_preserves_corner():
    path = SliderPath.build('B', [(0, 0), (100, 0), (100, 0), (100, 100)], 200)
    assert path.position_at(.5) == (100, 0)
    assert path.position_at(.75) == (100, 50)


def test_bezier_curvature_against_analytic_quadratic():
    path = SliderPath.build('B', [(0, 0), (50, 100), (100, 0)], 147.89428575445976)
    assert path.position_at(.5) == pytest.approx((50, 50), abs=.06)
    assert path.points[-1] == pytest.approx((100, 0), abs=.06)
    for x, y in path.points[:-1]:
        assert y == pytest.approx(2 * x * (1 - x / 100), abs=.06)


def test_collinear_bezier_backtracking_is_not_collapsed():
    path = SliderPath.build('B', [(0, 0), (100, 0), (0, 0)], 100)
    assert path.position_at(.5) == pytest.approx((50, 0), abs=.06)
    assert path.position_at(1) == pytest.approx((0, 0), abs=.06)


@pytest.mark.parametrize('controls', [
    [(0, 0), (50, 0), (100, 0)],
    [(0, 0), (50, 100), (100, 100), (150, 0)],
    [(0, 0), (0, 0), (100, 0)],
])
def test_degenerate_or_non_three_point_perfect_curve_uses_bezier(controls):
    perfect = SliderPath.build('P', controls, 100)
    bezier = SliderPath.build('B', controls, 100)
    assert perfect == bezier


def test_catmull_interpolates_knots_and_matches_interior_polynomial():
    controls = [(0, 0), (100, 100), (200, 0), (300, 100)]
    path = SliderPath.build('C', controls, 1000)
    # An extended path retains all original knots and interior subdivision points.
    assert all(knot in path.points for knot in controls)
    # Uniform Catmull at t=.25 between the middle two knots: x=125, y=84.375.
    assert any(np.linalg.norm(np.array(point) - (125, 84.375)) < .06 for point in path.points)


def test_repeated_catmull_knots_are_finite():
    path = SliderPath.build('C', [(0, 0), (50, 50), (50, 50), (100, 0)], 200)
    assert np.isfinite(path.points).all()
    assert np.all(np.diff(path.distances) > 0)


def test_curved_repeats_and_time_shift_share_geometry():
    path = SliderPath.build('P', [(100, 150), (150, 100), (200, 150)], 50 * pi)
    obj = HitObject(100, 150, 1000, 2, 2000, spans=2, path=path)
    assert obj.position_at(2000) == path.points[-1]
    assert obj.position_at(3000) == (100, 150)
    assert obj.position_at(1500) == obj.position_at(2500)
    shifted = replace(obj, time=2000)
    assert shifted.path is obj.path
    assert shifted.position_at(2500) == obj.position_at(1500)


@pytest.mark.parametrize('controls,length', [([(0, 0), (0, 0)], 10),
    ([(0, 0), (1, 1)], 0), ([(0, 0), (1, float('nan'))], 10),
    ([(0, 0), (1, 1)], float('inf'))])
def test_invalid_geometry_is_rejected(controls, length):
    with pytest.raises(ValueError):
        SliderPath.build('L', controls, length)


@pytest.mark.parametrize('hz', [60, 120, 240])
@pytest.mark.parametrize('beatmap,expected', [(MAP, 106), (CURVES, 4)])
def test_full_map_and_all_curve_fixture_are_attainable(hz, beatmap, expected):
    env = osuAimEnv(object_mode='mixed', beatmap_path=beatmap, sim_hz=hz)
    def checked_policy(obs, env):
        assert env.observation_space.contains(obs)
        return scripted_action(obs, env)
    result = evaluate(checked_policy, [7, 10001], env)
    assert result['targets'] == 2 * expected
    assert result['hits'] == result['targets']
    assert result['accuracy'] == 1
    assert result['bad_clicks'] == 0


def test_full_map_keeps_all_sliders_and_reachable_off_field_arcs():
    objects = load_hit_objects(MAP)
    sliders = [obj for obj in objects if obj.kind == 'slider']
    assert len(sliders) == 63
    assert any(x < 0 or y < 0 for obj in sliders for x, y in obj.path.points)
    env = osuAimEnv(object_mode='mixed', beatmap_path=MAP)
    env.reset(seed=0)
    assert env.interface_version == 'mixed-v3'


def test_unreachable_paths_are_rejected_by_environment(tmp_path):
    source = (ROOT / 'fixtures/mixed.osu').read_text()
    path = tmp_path / 'unreachable.osu'
    path.write_text(source.replace('L|290:150,2,140', 'L|900:150,2,800'))
    assert load_hit_objects(path)[1].end_position == (950, 150)
    with pytest.raises(ValueError, match='outside cursor reach'):
        osuAimEnv(object_mode='mixed', beatmap_path=path).reset()


def test_early_window_overlap_retains_current_object_priority(tmp_path):
    path = tmp_path / 'windows.osu'
    header = CURVES.read_text().split('[HitObjects]')[0]
    path.write_text(header + '[HitObjects]\n100,100,1000,1,0\n100,100,1200,1,0\n')
    env = osuAimEnv(object_mode='mixed', beatmap_path=path)
    env.reset(seed=0)
    env.cursor[:] = 100
    env.curr_step = 62
    env.current_time = 62 * env.dt
    _, _, _, _, info = env.step(np.array([0, 0, 1], dtype=np.float32))
    assert info['hits'] == 1
    assert info['target_index'] == 1
    assert info['judgements'][0]['index'] == 0
