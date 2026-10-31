from bisect import bisect_left
from dataclasses import dataclass
from math import acos, atan2, ceil, cos, hypot, isfinite, pi, sin

import numpy as np


@dataclass(frozen=True)
class SliderPath:
    points: tuple[tuple[float, float], ...]
    distances: tuple[float, ...]

    @classmethod
    def build(cls, curve_type, control_points, length):
        controls = np.asarray(control_points, dtype=np.float64)
        if (controls.ndim != 2 or controls.shape[1] != 2 or len(controls) < 2
                or not np.isfinite(controls).all() or not isfinite(length) or length <= 0):
            raise ValueError("Invalid slider geometry or length")
        if len(controls) > 256:
            raise ValueError("Slider exceeds the supported 256 control points")
        if curve_type == "L":
            raw = controls
        elif curve_type == "P":
            raw = _arc(controls)
        elif curve_type == "B":
            raw = _bezier_segments(controls)
        elif curve_type == "C":
            raw = _catmull(controls)
        else:
            raise ValueError(f"Unsupported slider curve type: {curve_type}")
        points = [tuple(map(float, raw[0]))]
        distances = [0.0]
        for point in raw[1:]:
            point = tuple(map(float, point))
            previous = points[-1]
            segment = hypot(point[0] - previous[0], point[1] - previous[1])
            if segment <= 1e-10:
                continue
            remaining = length - distances[-1]
            if segment >= remaining:
                points.append(tuple(a + (b - a) * remaining / segment for a, b in zip(previous, point)))
                distances.append(float(length))
                break
            points.append(point)
            distances.append(distances[-1] + segment)
        if len(points) < 2:
            raise ValueError("Slider geometry has no nonzero direction")
        if distances[-1] < length:
            # Extend along the final nonzero chord, as for the sampled curve.
            before, last = points[-2:]
            segment = distances[-1] - distances[-2]
            extra = length - distances[-1]
            points.append(tuple(b + (b - a) * extra / segment for a, b in zip(before, last)))
            distances.append(float(length))
        return cls(tuple(points), tuple(distances))

    def position_at(self, progress):
        """Position at a clamped fraction of total travelled distance."""
        if not isfinite(progress):
            raise ValueError("Path progress must be finite")
        if progress <= 0:
            return self.points[0]
        if progress >= 1:
            return self.points[-1]
        distance = progress * self.distances[-1]
        index = bisect_left(self.distances, distance)
        fraction = ((distance - self.distances[index - 1]) /
                    (self.distances[index] - self.distances[index - 1]))
        return tuple(a + (b - a) * fraction for a, b in
                     zip(self.points[index - 1], self.points[index]))


def _bezier(controls):
    # Adaptive de Casteljau subdivision. The second-difference bound controls
    # deviation from linear interpolation, including collinear backtracking.
    result = [controls[0]]
    pending = [(controls, 0)]
    while pending:
        points, depth = pending.pop()
        degree = len(points) - 1
        differences = np.diff(points, n=2, axis=0)
        bound = (degree * (degree - 1) / 8 * np.max(np.linalg.norm(differences, axis=1))
                 if degree > 1 else 0)
        if bound <= 0.05:
            result.append(points[-1])
            continue
        if depth >= 20 or len(result) + len(pending) > 65536:
            raise ValueError("Slider curve exceeds supported approximation complexity")
        work = points.copy()
        left, right = [work[0].copy()], [work[-1].copy()]
        while len(work) > 1:
            work = (work[:-1] + work[1:]) / 2
            left.append(work[0].copy())
            right.append(work[-1].copy())
        pending.append((np.array(right[::-1]), depth + 1))
        pending.append((np.array(left), depth + 1))
    return result


def _bezier_segments(controls):
    result = []
    start = 0
    for index in range(len(controls) - 1):
        if np.array_equal(controls[index], controls[index + 1]):
            result.extend(_bezier(controls[start:index + 1]))
            start = index + 1
    result.extend(_bezier(controls[start:]))
    return result


def _arc(controls):
    # Non-three-point and degenerate perfect curves fall back to Bezier.
    if len(controls) != 3:
        return _bezier_segments(controls)
    a, b, c = controls
    u, v = b - a, c - a
    determinant = u[0] * v[1] - u[1] * v[0]
    scale = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1.0)
    if abs(determinant) <= 1e-10 * scale:
        return _bezier_segments(controls)
    center = a + np.array([v[1] * (u @ u) - u[1] * (v @ v),
                           u[0] * (v @ v) - v[0] * (u @ u)]) / (2 * determinant)
    radius = float(np.linalg.norm(a - center))
    angles = [atan2(point[1] - center[1], point[0] - center[0]) for point in controls]
    sweep = (angles[2] - angles[0]) % (2 * pi)
    if (angles[1] - angles[0]) % (2 * pi) > sweep:
        sweep -= 2 * pi
    # At most 0.05 pixels of sagitta error per chord.
    angle_step = 2 * acos(max(-1.0, min(1.0, 1 - 0.05 / radius)))
    count = max(1, ceil(abs(sweep) / max(angle_step, 1e-12)))
    if count > 65536:
        raise ValueError("Circular slider exceeds supported approximation complexity")
    points = [center + radius * np.array([cos(angles[0] + sweep * i / count),
                                         sin(angles[0] + sweep * i / count)]) for i in range(count + 1)]
    points[0], points[-1] = a, c
    return points


def _catmull(controls):
    # osu!'s current framework uses uniform Catmull-Rom, with a duplicated first
    # knot and extrapolated last knot. Express each span as a cubic Bezier so
    # the same adaptive approximation bound applies to all polynomial paths.
    result = []
    for i in range(len(controls) - 1):
        previous = controls[i - 1] if i else controls[i]
        start, end = controls[i:i + 2]
        following = controls[i + 2] if i + 2 < len(controls) else 2 * end - start
        cubic = np.array([start, start + (end - previous) / 6,
                          end - (following - start) / 6, end])
        result.extend(_bezier(cubic))
    return result
