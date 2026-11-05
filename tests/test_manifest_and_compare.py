"""Checks on manifest capture and comparison maths.

The point of these is that a manifest must never silently claim more than the run
proved, and that the comparison numbers behave correctly at their edges.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsafenet_cf import compare, manifest  # noqa: E402


# --- manifest -------------------------------------------------------------------


def test_gpu_snapshot_degrades_without_torch_instead_of_raising():
    snap = manifest.gpu_snapshot()
    assert "available" in snap
    if not snap["available"]:
        assert snap["reason"]


def test_package_versions_reports_missing_as_none():
    versions = manifest.package_versions(("numpy", "definitely_not_a_package"))
    assert versions["definitely_not_a_package"] is None
    assert versions["numpy"] == np.__version__


def test_peak_memory_is_none_without_a_gpu():
    # Must not raise on a machine with no CUDA; the notebook calls it either way.
    assert manifest.peak_memory_gb() in (None,) or isinstance(
        manifest.peak_memory_gb(), float
    )


def test_manifest_never_claims_safety_or_ld_vice():
    m = manifest.build_run_manifest(
        stage="prompt_guided_edit",
        environment={"gpu": {"available": False}},
        settings={"seed": 1},
        inputs={},
        outputs={},
        timing={},
        memory={},
    )
    assert m["claims"]["is_ld_vice_reproduction"] is False
    assert m["claims"]["safety_verified"] is False
    assert m["claims"]["classifier_evaluated"] is False
    assert "candidate" in m["claims"]["status"]
    assert m["schema"] == "unsafenet-cf/run-manifest/1"


def test_write_manifest_round_trips(tmp_path):
    import json

    m = manifest.build_run_manifest("vae_reconstruction", {}, {}, {}, {}, {}, {})
    path = manifest.write_manifest(m, tmp_path / "nested" / "manifest.json")
    assert path.exists()
    assert json.loads(path.read_text())["stage"] == "vae_reconstruction"


def test_sha256_file_matches_known_value(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert manifest.sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --- compare --------------------------------------------------------------------


def _clip(value, t=4, h=8, w=8):
    return np.full((t, h, w, 3), value, dtype=np.uint8)


def test_psnr_of_identical_clips_is_infinite():
    a = _clip(100)
    assert compare.psnr(a, a) == float("inf")


def test_psnr_of_known_difference():
    a = _clip(0)
    b = _clip(255)
    # mse = 255^2, so psnr = 10*log10(255^2/255^2) = 0
    assert compare.psnr(a, b) == pytest.approx(0.0, abs=1e-9)


def test_psnr_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare.psnr(_clip(0, t=4), _clip(0, t=5))


def test_per_frame_psnr_exposes_drift():
    a = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    b = a.copy()
    b[2] = 255  # only the last frame differs
    values = compare.per_frame_psnr(a, b)
    assert values[0] == float("inf")
    assert values[1] == float("inf")
    assert values[2] == pytest.approx(0.0, abs=1e-9)


def test_mean_abs_diff():
    assert compare.mean_abs_diff(_clip(10), _clip(14)) == pytest.approx(4.0)


def test_pick_indices_includes_first_and_last():
    assert compare.pick_indices(49, 4) == [0, 16, 32, 48]
    assert compare.pick_indices(49, 1) == [0]
    assert compare.pick_indices(2, 4) == [0, 1]


def test_to_array_rejects_wrong_shape():
    with pytest.raises(ValueError, match="T, H, W, 3"):
        compare.to_array([np.zeros((8, 8), dtype=np.uint8)])


def test_comparison_grid_layout(tmp_path):
    pytest.importorskip("PIL")
    a = _clip(10, t=4, h=6, w=5)
    b = _clip(20, t=4, h=6, w=5)
    result = compare.comparison_grid(
        [("source", a), ("reconstruction", b)],
        indices=[0, 3],
        out_path=tmp_path / "grid.png",
    )
    assert result["order"] == ["source", "reconstruction"]
    # 2 rows of height 6, 2 columns of width 5
    assert result["image"].size == (10, 12)
    assert (tmp_path / "grid.png").exists()


def test_comparison_grid_rejects_mismatched_frame_sizes():
    with pytest.raises(ValueError, match="differ in frame size"):
        compare.comparison_grid(
            [("a", _clip(1, h=6, w=5)), ("b", _clip(1, h=8, w=5))]
        )


def test_comparison_grid_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="outside"):
        compare.comparison_grid([("a", _clip(1, t=4))], indices=[0, 9])
