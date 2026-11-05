"""Checks on the preprocessing maths that must hold before any GPU time is spent.

These are deliberately about the things that silently corrupt a run: a crop that
leaves the frame, a temporal shape the VAE cannot take, a segment that runs off
the end of the clip, and an ffmpeg filter whose stages are in the wrong order.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsafenet_cf import preprocess as pp  # noqa: E402


# --- checkpoint constants match the published configs ---------------------------


def test_constants_match_checkpoint_config():
    # THUDM/CogVideoX-2b vae/config.json and transformer/config.json
    assert (pp.TARGET_WIDTH, pp.TARGET_HEIGHT) == (720, 480)
    assert pp.SAMPLE_FRAMES == 49
    assert pp.TEMPORAL_COMPRESSION_RATIO == 4
    assert pp.LATENT_CHANNELS == 16
    assert pp.VAE_SCALING_FACTOR == 1.15258426
    # transformer sample_height=60, sample_width=90 at spatial ratio 8
    assert pp.TARGET_HEIGHT // pp.SPATIAL_COMPRESSION_RATIO == 60
    assert pp.TARGET_WIDTH // pp.SPATIAL_COMPRESSION_RATIO == 90


# --- temporal shape -------------------------------------------------------------


def test_latent_frame_count_for_sample_frames():
    assert pp.latent_frame_count(49) == 13


@pytest.mark.parametrize("n", [1, 5, 9, 49, 81])
def test_latent_frame_count_accepts_valid_shapes(n):
    assert pp.latent_frame_count(n) == (n - 1) // 4 + 1


@pytest.mark.parametrize("n", [2, 3, 4, 48, 50])
def test_latent_frame_count_rejects_bad_temporal_shape(n):
    with pytest.raises(pp.PreprocessError):
        pp.latent_frame_count(n)


def test_validate_frame_count_strict_rejects_non_sample_frames():
    # 81 is temporally valid but is not what CogVideoX-2b was trained at.
    pp.latent_frame_count(81)
    with pytest.raises(pp.PreprocessError, match="sample_frames"):
        pp.validate_frame_count(81)
    pp.validate_frame_count(81, strict=False)  # explicit opt-out is allowed


def test_validate_frame_count_accepts_49():
    pp.validate_frame_count(49)


# --- resolution -----------------------------------------------------------------


def test_validate_resolution_rejects_non_multiple_of_eight():
    with pytest.raises(pp.PreprocessError, match="divisible"):
        pp.validate_resolution(722, 480, strict=False)


def test_validate_resolution_strict_rejects_other_sizes():
    pp.validate_resolution(1280, 720, strict=False)
    with pytest.raises(pp.PreprocessError):
        pp.validate_resolution(1280, 720)


# --- crop geometry --------------------------------------------------------------


def test_crop_from_1920x1080_to_3_2_is_1620_wide():
    box = pp.aspect_preserving_crop(1920, 1080)
    assert (box.width, box.height) == (1620, 1080)
    assert box.x == 150  # centred by default
    assert box.y == 0


def test_explicit_x_offset_keeps_the_walkway():
    # The selected clip uses x=300 so the right-hand green walkway survives.
    box = pp.aspect_preserving_crop(1920, 1080, x=300)
    assert (box.width, box.height, box.x, box.y) == (1620, 1080, 300, 0)
    assert box.x + box.width == 1920  # crop reaches the right edge of the source


def test_crop_outside_source_is_rejected():
    with pytest.raises(pp.PreprocessError, match="outside"):
        pp.aspect_preserving_crop(1920, 1080, x=400)


def test_crop_dimensions_are_even_for_yuv420p():
    for w, h in [(1921, 1081), (1365, 767), (640, 361)]:
        box = pp.aspect_preserving_crop(w, h)
        assert box.width % 2 == 0
        assert box.height % 2 == 0


def test_crop_filter_string():
    box = pp.CropBox(1620, 1080, 300, 0)
    assert box.as_filter() == "crop=1620:1080:300:0"


# --- temporal segment -----------------------------------------------------------


def test_frame_indices_for_the_selected_clip():
    idx = pp.frame_indices(0, 3, 49)
    assert len(idx) == 49
    assert idx[0] == 0
    assert idx[-1] == 144


def test_segment_fits_the_262_frame_source():
    idx = pp.validate_segment(0, 3, 49, 262)
    assert idx[-1] == 144 < 262


def test_segment_running_past_the_end_is_rejected():
    with pytest.raises(pp.PreprocessError, match="only 100 frames"):
        pp.validate_segment(0, 3, 49, 100)


def test_effective_source_fps():
    assert pp.effective_source_fps(25.0, 3) == pytest.approx(8.3333, rel=1e-3)


# --- ffmpeg command -------------------------------------------------------------


def _spec():
    return pp.PreprocessSpec(
        source="data/raw/0_tr143.mp4",
        crop=pp.CropBox(1620, 1080, 300, 0),
        start_frame=0,
        frame_step=3,
    )


def test_filter_stages_are_ordered_select_crop_scale_setpts():
    cmd = pp.build_ffmpeg_command(_spec(), "out.mp4", ffmpeg="ffmpeg")
    vf = cmd[cmd.index("-vf") + 1]
    positions = [vf.index(part) for part in ("select=", "crop=", "scale=", "setpts=")]
    assert positions == sorted(positions), vf


def test_select_expression_covers_exactly_the_requested_frames():
    cmd = pp.build_ffmpeg_command(_spec(), "out.mp4", ffmpeg="ffmpeg")
    vf = cmd[cmd.index("-vf") + 1]
    # last selected source frame is 0 + 48*3 = 144
    assert r"gte(n\,0)" in vf
    assert r"lte(n\,144)" in vf
    assert r"not(mod(n-0\,3))" in vf


def test_command_requests_the_exact_frame_count_and_rate():
    cmd = pp.build_ffmpeg_command(_spec(), "out.mp4", ffmpeg="ffmpeg")
    assert cmd[cmd.index("-frames:v") + 1] == "49"
    assert cmd[cmd.index("-r") + 1] == "8"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[-1] == "out.mp4"


def test_command_rejects_an_invalid_frame_count_before_running():
    bad = pp.PreprocessSpec(
        source="x.mp4",
        crop=pp.CropBox(1620, 1080, 300, 0),
        start_frame=0,
        frame_step=3,
        num_frames=50,
    )
    with pytest.raises(pp.PreprocessError):
        pp.build_ffmpeg_command(bad, "out.mp4")


# --- spec serialisation ---------------------------------------------------------


def test_spec_to_dict_records_derived_latent_shape():
    d = _spec().to_dict()
    assert d["latent_frames"] == 13
    assert (d["latent_height"], d["latent_width"]) == (60, 90)
    assert d["selected_frame_indices"][-1] == 144
    assert len(d["selected_frame_indices"]) == 49


def test_shipped_clip_config_loads_and_is_valid():
    root = Path(__file__).resolve().parents[1]
    spec = pp.load_spec(root / "configs" / "clip_0_tr143.json")
    assert spec.num_frames == 49
    assert (spec.width, spec.height) == (720, 480)
    assert spec.crop == pp.CropBox(1620, 1080, 300, 0)
    # the config must describe a segment that exists in the 262-frame source
    pp.validate_segment(spec.start_frame, spec.frame_step, spec.num_frames, 262)
    # and a command must be buildable from it
    pp.build_ffmpeg_command(spec, "out.mp4", ffmpeg="ffmpeg")


def test_clip_config_sha256_matches_provenance():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "clip_0_tr143.json").read_text())
    prov = json.loads((root / "data" / "raw" / "PROVENANCE.json").read_text())
    entry = next(
        f for f in prov["downloaded"] if f["filename"] == "0_tr143.mp4"
    )
    assert cfg["source_sha256"] == entry["sha256"]
    assert cfg["unsafe_label"]["name"] == prov["dataset"]["classes"]["0"]["name"]
    assert cfg["target_safe_label"]["name"] == prov["dataset"]["classes"]["4"]["name"]


# --- ffmpeg report parsing ------------------------------------------------------


REPORT = """
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuvj420p(pc, progressive), 1920x1080, 2386 kb/s, 25.07 fps, 25 tbr, 19672 tbn (default)
  Stream #0:0 -> #0:0 (h264 (native) -> wrapped_avframe (native))
frame=  262 fps=0.0 q=-0.0 Lsize=N/A time=00:00:10.44 bitrate=N/A speed=54.3x
"""


def test_parse_ffmpeg_report_extracts_real_values():
    info = pp._parse_ffmpeg_report(REPORT)
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["frames"] == 262
    assert info["codec"] == "h264"
    assert info["pix_fmt"] == "yuvj420p"
    assert info["duration_s"] == pytest.approx(10.44)
    assert info["fps"] == pytest.approx(25.07)
