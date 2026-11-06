"""Tests for the panel-cover registration and compositing geometry.

Every test here builds a case whose answer is known before the code runs: a known shift, a
known transform, a known alpha. That is the only way these helpers can be checked without
the dataset, and the geometry is where a silent error would be most expensive later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsafenet_cf import panelpatch as pp  # noqa: E402


def textured(h: int = 64, w: int = 80, seed: int = 7) -> np.ndarray:
    """A textured greyscale plane. Smooth enough to interpolate, not flat anywhere."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (h, w))
    return np.clip(pp.box_mean(base, 5), 0, 255)


# --- normalised cross-correlation ------------------------------------------------


def test_ncc_of_a_patch_with_itself_is_one():
    a = textured()
    assert pp.ncc(a, a) == pytest.approx(1.0)


def test_ncc_is_invariant_to_brightness_and_contrast():
    a = textured()
    assert pp.ncc(a, 0.5 * a + 30) == pytest.approx(1.0)


def test_ncc_of_a_flat_patch_is_zero_not_an_error():
    assert pp.ncc(np.zeros((8, 8)), textured(8, 8)) == 0.0


# --- landmark offsets ------------------------------------------------------------


def test_patch_offsets_recovers_a_known_shift():
    src = textured(80, 100)
    dx, dy = 5, -3
    dst = np.roll(np.roll(src, dx, axis=1), dy, axis=0)
    res = pp.patch_offsets(src, dst, [(30, 30, 20, 20), (55, 40, 16, 16)], search=8)
    assert all(r["ok"] for r in res)
    for r in res:
        assert r["offset_xy"] == [dx, dy]
        assert r["ncc"] > 0.99


def test_patch_offsets_reports_the_peak_margin():
    src = textured(80, 100)
    dst = np.roll(src, 4, axis=1)
    r = pp.patch_offsets(src, dst, [(30, 30, 24, 24)], search=8)[0]
    assert r["margin_over_second"] > 0.0


def test_patch_offsets_rejects_a_patch_outside_the_image():
    with pytest.raises(ValueError, match="outside"):
        pp.patch_offsets(textured(40, 40), textured(40, 40), [(30, 30, 20, 20)])


# --- transform fitting -----------------------------------------------------------


def test_translation_fit_is_exact_on_a_translated_set():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [7.0, 3.0]])
    dst = src + np.array([3.5, -2.0])
    fit = pp.estimate_translation(src, dst)
    assert fit["residual_px"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert pp.apply_transform(fit["_matrix"], [[1.0, 1.0]])[0] == pytest.approx([4.5, -1.0])


def test_similarity_fit_recovers_scale_and_rotation():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [4.0, 6.0]])
    theta, scale, t = np.deg2rad(20.0), 1.3, np.array([5.0, -4.0])
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = src @ (scale * r).T + t
    fit = pp.estimate_similarity(src, dst)
    assert fit["residual_px"]["max"] == pytest.approx(0.0, abs=1e-8)
    assert fit["degrees_of_freedom"] == 2
    m = np.asarray(fit["_matrix"])
    assert np.hypot(m[0, 0], m[1, 0]) == pytest.approx(scale)


def test_affine_fit_recovers_a_shear():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    a = np.array([[1.2, 0.3], [0.0, 0.9]])
    dst = src @ a.T + np.array([2.0, 1.0])
    fit = pp.estimate_affine(src, dst)
    assert fit["residual_px"]["max"] == pytest.approx(0.0, abs=1e-8)


def test_homography_fit_recovers_a_projective_warp():
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0], [50.0, 40.0]])
    h = np.array([[1.1, 0.05, 4.0], [0.02, 0.95, -3.0], [0.0008, 0.0004, 1.0]])
    dst = pp.apply_transform(h, src)
    fit = pp.estimate_homography(src, dst)
    assert fit["residual_px"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert fit["degrees_of_freedom"] == 1


def test_an_exact_fit_reports_zero_degrees_of_freedom():
    src = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    fit = pp.estimate_affine(src, src * 2)
    assert fit["degrees_of_freedom"] == 0
    assert "not evidence of accuracy" in fit["_residual_note"]


@pytest.mark.parametrize("fn,n", [(pp.estimate_similarity, 1), (pp.estimate_affine, 2),
                                  (pp.estimate_homography, 3)])
def test_fits_refuse_too_few_points(fn, n):
    pts = np.zeros((n, 2))
    with pytest.raises(ValueError):
        fn(pts, pts)


# --- warping ---------------------------------------------------------------------


def test_warp_reproduces_an_integer_translation():
    img = textured(50, 60)
    m = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    out, valid = pp.warp(img, m)
    assert out[20 + 2, 20 + 4] == pytest.approx(img[20, 20])
    # The first four columns and two rows sample from outside the source.
    assert not valid[0, 0]
    assert valid[20 + 2, 20 + 4]


def test_warp_of_the_identity_returns_the_image():
    img = textured(32, 40)
    out, valid = pp.warp(img, np.eye(3))
    assert np.abs(out - img).max() == pytest.approx(0.0, abs=1e-9)
    assert valid.all()


def test_warp_keeps_three_channels():
    img = np.stack([textured(24, 30)] * 3, axis=2)
    out, _ = pp.warp(img, np.eye(3))
    assert out.shape == img.shape


# --- masks, feathering, compositing ----------------------------------------------


def test_box_mean_preserves_a_constant_plane():
    plane = np.full((20, 25), 7.0)
    assert np.abs(pp.box_mean(plane, 5) - 7.0).max() == pytest.approx(0.0, abs=1e-9)


def test_feather_with_zero_radius_is_a_hard_alpha():
    mask = np.zeros((10, 10), bool)
    mask[3:7, 3:7] = True
    a = pp.feather(mask, 0)
    assert set(np.unique(a)) == {0.0, 1.0}


def test_feather_softens_only_near_the_edge():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    a = pp.feather(mask, 2)
    assert a[10, 10] == pytest.approx(1.0)      # deep inside stays opaque
    assert 0.0 < a[5, 10] < 1.0                 # on the edge is partial
    assert a[0, 0] == pytest.approx(0.0)        # far outside stays clear


def test_composite_with_alpha_zero_and_one():
    base = np.full((8, 8, 3), 10, np.uint8)
    donor = np.full((8, 8, 3), 200, np.uint8)
    assert (pp.composite(base, donor, np.zeros((8, 8))) == 10).all()
    assert (pp.composite(base, donor, np.ones((8, 8))) == 200).all()


def test_composite_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="differ"):
        pp.composite(np.zeros((4, 4, 3)), np.zeros((5, 4, 3)), np.ones((4, 4)))


def test_unchanged_outside_mask_is_clean_for_a_hard_composite():
    base = np.full((12, 12, 3), 30, np.uint8)
    donor = np.full((12, 12, 3), 240, np.uint8)
    mask = np.zeros((12, 12), bool)
    mask[4:8, 4:8] = True
    out = pp.composite(base, donor, pp.feather(mask, 0))
    rep = pp.unchanged_outside_mask(base, out, mask)
    assert rep["clean"] and rep["pixels_changed_outside_mask"] == 0


def test_unchanged_outside_mask_catches_a_leak():
    base = np.full((12, 12, 3), 30, np.uint8)
    out = base.copy()
    out[0, 0] = 99
    mask = np.zeros((12, 12), bool)
    mask[4:8, 4:8] = True
    rep = pp.unchanged_outside_mask(base, out, mask)
    assert not rep["clean"]
    assert rep["max_abs_change_outside_mask"] == 69


# --- seam and stability measures --------------------------------------------------


def test_ring_outside_is_disjoint_from_the_mask():
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    outer = pp.ring(mask, 2, outside=True)
    assert not (outer & mask).any()
    assert outer.any()


def test_ring_inside_lies_within_the_mask():
    mask = np.zeros((30, 30), bool)
    mask[10:20, 10:20] = True
    inner = pp.ring(mask, 2, outside=False)
    assert (inner & mask).sum() == inner.sum()
    assert inner.any()


def test_seam_contrast_is_near_zero_on_a_uniform_image():
    img = np.full((40, 60), 100.0)
    mask = np.zeros((40, 60), bool)
    mask[12:24, 12:24] = True
    rep = pp.seam_contrast(img, mask, width=2)
    assert rep["usable"]
    assert rep["boundary_step"] == pytest.approx(0.0, abs=1e-9)


def test_seam_contrast_sees_a_step_at_the_mask_edge():
    img = np.full((40, 60), 100.0)
    mask = np.zeros((40, 60), bool)
    mask[12:24, 12:24] = True
    img[mask] = 160.0
    rep = pp.seam_contrast(img, mask, width=2)
    assert rep["boundary_step"] > 50.0


def test_region_stability_is_zero_for_a_static_clip():
    frames = np.repeat(textured(20, 24)[None, ..., None], 6, axis=0)
    frames = np.repeat(frames, 3, axis=3).astype(np.uint8)
    mask = np.zeros((20, 24), bool)
    mask[5:15, 5:15] = True
    rep = pp.region_stability(frames, mask)
    assert rep["mad_in_mask"] == pytest.approx(0.0, abs=1e-9)
    assert rep["frames"] == 6


def test_region_stability_reports_movement_inside_the_mask():
    frames = np.zeros((7, 20, 24, 3), np.uint8)
    for t in range(7):
        frames[t, 8:12, 8:12] = 10 + 30 * t
    mask = np.zeros((20, 24), bool)
    mask[8:12, 8:12] = True
    rep = pp.region_stability(frames, mask)
    assert rep["mad_in_mask"] > rep["mad_whole_frame"]
