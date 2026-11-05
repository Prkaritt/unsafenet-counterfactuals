"""Checks on the relocation geometry, run before any GPU time is spent on it.

The point of these is that a path check must never report a plausible path when the path
teleports or crosses machinery, that compositing must land a silhouette where it was asked
to and say so when it could not, and that the background estimate must admit which pixels
it has no evidence for.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsafenet_cf import manualpath as mp  # noqa: E402


# --- regions --------------------------------------------------------------------


def test_connected_components_separates_two_blobs_and_drops_speckle():
    m = np.zeros((40, 40), bool)
    m[2:10, 2:10] = True      # 64 px
    m[20:32, 20:34] = True    # 168 px
    m[35, 35] = True          # speckle, must be dropped
    comps = mp.connected_components(m, min_area=10)
    assert len(comps) == 2
    assert [c["area"] for c in comps] == [168, 64]
    assert comps[0]["bbox"] == [20, 20, 33, 31]


def test_connected_components_joins_diagonal_neighbours():
    m = np.zeros((10, 10), bool)
    m[2:5, 2:5] = True
    m[5, 5] = True  # touches the block only at a corner
    comps = mp.connected_components(m, min_area=1)
    assert len(comps) == 1, "8-connectivity must treat a diagonal touch as one component"


def test_convex_hull_of_a_square_is_its_corners():
    pts = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5], [3, 7]], float)
    hull = mp.convex_hull(pts)
    assert len(hull) == 4
    corners = {tuple(p) for p in hull}
    assert corners == {(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)}


def test_point_in_polygon_inside_and_outside():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert mp.point_in_polygon(5, 5, poly)
    assert not mp.point_in_polygon(15, 5, poly)
    assert not mp.point_in_polygon(5, -1, poly)


def test_polygon_mask_area_matches_the_polygon():
    poly = np.array([[2, 2], [8, 2], [8, 6], [2, 6]], float)
    m = mp.polygon_mask(poly, (12, 12))
    # 7 columns by 5 rows inclusive of both edges, allowing one row or column of
    # rasterisation slack.
    assert 24 <= m.sum() <= 40
    assert m[4, 5]
    assert not m[10, 10]


# --- perspective ----------------------------------------------------------------


def test_height_fit_recovers_a_known_line():
    y = list(np.linspace(300, 400, 20))
    h = [0.5 * v + 10 for v in y]
    fit = mp.fit_height_vs_y(y, h)
    assert fit["usable"]
    assert fit["slope_px_height_per_px_row"] == pytest.approx(0.5, abs=1e-3)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_height_fit_refuses_when_there_is_no_spread():
    fit = mp.fit_height_vs_y([400.0] * 10, [120.0] * 10)
    assert not fit["usable"]


def test_scale_at_refuses_to_extrapolate_outside_the_measured_rows():
    fit = mp.fit_height_vs_y(list(np.linspace(300, 400, 20)),
                             [0.5 * v + 10 for v in np.linspace(300, 400, 20)])
    inside = mp.scale_at(fit, 350.0, reference_height=100.0)
    outside = mp.scale_at(fit, 470.0, reference_height=100.0)
    assert inside == pytest.approx(1.85, abs=0.01)
    assert outside == 1.0, "an unsupported row must not silently rescale the worker"


# --- paths ----------------------------------------------------------------------


def test_interpolate_path_leaves_pre_intervention_frames_empty():
    kf = [{"frame": 4, "x": 0.0, "y": 100.0}, {"frame": 8, "x": 40.0, "y": 100.0}]
    path = mp.interpolate_path(kf, 12)
    assert path[0] is None and path[3] is None
    assert path[4]["x"] == pytest.approx(0.0)
    assert path[6]["x"] == pytest.approx(20.0)
    assert path[8]["x"] == pytest.approx(40.0)
    assert path[11]["x"] == pytest.approx(40.0), "after the last keyframe the path holds"


def test_path_checks_flags_a_teleport_on_lateral_step_only():
    lane = np.array([[100, 90], [140, 90], [140, 110], [100, 110]], float)
    path = [{"frame": 0, "x": 0.0, "y": 100.0},
            {"frame": 1, "x": 5.0, "y": 100.0},
            {"frame": 2, "x": 120.0, "y": 100.0}]
    feet = [(0.0, 100.0), (5.0, 100.0), (10.0, 100.0)]
    out = mp.path_checks(path, feet, lane, {}, speed_limit_px=20.0)
    assert out["teleports"]
    assert out["steps_over_limit"][0]["frame"] == 2
    assert out["max_lateral_step_px"] == pytest.approx(115.0)


def test_path_checks_does_not_flag_vertical_motion_copied_from_the_source():
    """Under follow_source the vertical part is the worker's own jitter, not the edit."""
    lane = np.array([[100, 90], [140, 90], [140, 130], [100, 130]], float)
    path = [{"frame": 0, "x": 100.0, "y": 100.0},
            {"frame": 1, "x": 105.0, "y": 128.0}]
    feet = [(100.0, 100.0), (100.0, 128.0)]
    out = mp.path_checks(path, feet, lane, {}, speed_limit_px=10.0)
    assert not out["teleports"], "a 5 px lateral step is inside a 10 px limit"
    assert out["max_total_step_px"] > out["speed_limit_px"]


def test_path_checks_reports_machinery_crossing_and_lane_arrival():
    lane = np.array([[200, 90], [240, 90], [240, 110], [200, 110]], float)
    machine = np.array([[100, 90], [150, 90], [150, 110], [100, 110]], float)
    path = [{"frame": 0, "x": 90.0, "y": 100.0},
            {"frame": 1, "x": 120.0, "y": 100.0},
            {"frame": 2, "x": 220.0, "y": 100.0}]
    feet = [(90.0, 100.0)] * 3
    out = mp.path_checks(path, feet, lane, {"press": machine}, speed_limit_px=200.0)
    assert out["crosses_machinery"]
    assert out["frames_inside_obstacle"][0] == {"frame": 1, "obstacle": "press"}
    assert out["ends_inside_lane_polygon"]
    assert out["start_offset_from_source_foot_px"] == pytest.approx(0.0)


# --- compositing ----------------------------------------------------------------


def test_temporal_background_ignores_the_excluded_pixels():
    frames = np.zeros((5, 4, 4, 3), np.uint8)
    frames[:] = 10
    frames[2, 1, 1] = 200          # one bright intruder
    exclude = np.zeros((5, 4, 4), bool)
    exclude[2, 1, 1] = True
    bg, unknown = mp.temporal_background(frames, exclude)
    assert bg[1, 1].tolist() == [10, 10, 10]
    assert not unknown.any()


def test_temporal_background_flags_pixels_with_no_evidence():
    frames = np.full((3, 2, 2, 3), 7, np.uint8)
    exclude = np.zeros((3, 2, 2), bool)
    exclude[:, 0, 0] = True        # covered in every frame
    bg, unknown = mp.temporal_background(frames, exclude)
    assert unknown[0, 0]
    assert not unknown[1, 1]
    assert bg[0, 0].tolist() == [7, 7, 7], "the flagged pixel still gets a fallback value"


def test_paste_silhouette_puts_the_foot_point_where_asked():
    canvas = np.zeros((50, 50, 3), np.uint8)
    patch = np.full((10, 6, 3), 255, np.uint8)
    alpha = np.ones((10, 6), np.float32)
    res = mp.paste_silhouette(canvas, patch, alpha, (25.0, 30.0))
    assert res["pasted"] and not res["clipped"]
    assert canvas[29, 25].tolist() == [255, 255, 255], "bottom row sits on the foot row"
    assert canvas[20, 25].tolist() == [255, 255, 255], "top row is the 10th row above it"
    assert canvas[19, 25].tolist() == [0, 0, 0], "and nothing is drawn above that"
    assert canvas[30, 25].tolist() == [0, 0, 0], "nothing is drawn below the foot point"


def test_paste_silhouette_reports_when_it_falls_outside_the_frame():
    canvas = np.zeros((20, 20, 3), np.uint8)
    patch = np.full((5, 5, 3), 255, np.uint8)
    alpha = np.ones((5, 5), np.float32)
    res = mp.paste_silhouette(canvas, patch, alpha, (-50.0, 10.0))
    assert not res["pasted"]
    assert canvas.sum() == 0


def test_paste_silhouette_respects_alpha():
    canvas = np.zeros((20, 20, 3), np.uint8)
    patch = np.full((4, 4, 3), 255, np.uint8)
    alpha = np.zeros((4, 4), np.float32)
    alpha[0, 0] = 1.0
    mp.paste_silhouette(canvas, patch, alpha, (10.0, 10.0))
    assert canvas.sum() == 255 * 3, "only the one opaque pixel may be written"
