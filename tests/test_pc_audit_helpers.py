"""Tests for the audit helpers that decide where a panel might be.

These run without the dataset. Each case plants a difference whose position and size are
known, so a change in the smoothing, the blob threshold or the box arithmetic shows up as a
failed assertion rather than as a slightly wrong candidate region in a 148-clip report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pc_audit as A  # noqa: E402


def test_box_mean_preserves_a_constant_plane():
    plane = np.full((20, 25), 7.0)
    assert np.abs(A.box_mean(plane, 9) - 7.0).max() == pytest.approx(0.0, abs=1e-6)


def test_box_mean_forces_an_odd_window():
    plane = np.random.default_rng(1).normal(size=(12, 14))
    assert A.box_mean(plane, 4).shape == plane.shape


def test_resize_nn_keeps_channels_and_shape():
    assert A.resize_nn(np.zeros((10, 10), np.uint8), 5, 7).shape == (5, 7)
    assert A.resize_nn(np.zeros((10, 10, 3), np.uint8), 5, 5).shape == (5, 5, 3)


def test_corr_is_one_for_an_affine_rescaling():
    a = np.random.default_rng(0).normal(size=(30, 40))
    assert A.corr(a, a) == pytest.approx(1.0)
    assert A.corr(a, 2 * a + 5) == pytest.approx(1.0)


def test_corr_of_a_flat_plane_is_zero_not_an_error():
    a = np.random.default_rng(0).normal(size=(5, 5))
    assert A.corr(np.zeros((5, 5)), a) == 0.0


def test_difference_boxes_locates_a_planted_square():
    base = np.full((60, 80), 100.0)
    other = base.copy()
    other[20:35, 30:50] = 160.0
    boxes = A.difference_boxes(base, other, 18.0)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]["box_bg_scale"]
    # Smoothing spreads the edge outward by up to half the window, so the box is at least
    # as large as the plant and centred on it.
    assert x <= 30 and y <= 20
    assert x + w >= 50 and y + h >= 35
    assert boxes[0]["max_diff_in_box"] > 18.0


def test_difference_boxes_finds_nothing_between_identical_medians():
    base = np.full((40, 50), 90.0)
    assert A.difference_boxes(base, base.copy(), 18.0) == []


def test_difference_boxes_ignores_a_blob_below_the_minimum_area():
    base = np.full((40, 50), 90.0)
    other = base.copy()
    other[10:12, 10:12] = 200.0          # 4 px, far below MIN_BLOB
    assert A.difference_boxes(base, other, 18.0) == []


def test_difference_boxes_sorts_largest_region_first():
    base = np.full((80, 100), 100.0)
    other = base.copy()
    other[10:20, 10:20] = 170.0          # smaller
    other[40:65, 50:85] = 170.0          # larger
    boxes = A.difference_boxes(base, other, 18.0)
    assert len(boxes) == 2
    assert boxes[0]["area_px"] > boxes[1]["area_px"]


def test_heatmap_shape_and_range():
    diff = np.abs(np.random.default_rng(2).normal(size=(30, 40))) * 20
    hm = A.heatmap(diff)
    assert hm.shape == (30, 40, 3)
    assert 0 <= hm.min() and hm.max() <= 255


def test_label_strip_draws_something():
    s = A.label_strip(320, "2_tr100 f12")
    assert s.shape == (16, 320, 3)
    assert s.max() > 0
