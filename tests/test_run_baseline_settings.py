"""Checks on the generation-settings maths in scripts/run_baseline.py.

Two things are easy to get silently wrong and both corrupt the record rather than
crashing the run: the number of denoising steps the pipeline actually executes, and the
fallback order when a --sweep value omits a key.

Importing the script must not import torch or diffusers, which are absent on a login
node, so these tests also pin that the module stays importable without a GPU stack.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "run_baseline", REPO / "scripts" / "run_baseline.py")
run_baseline = importlib.util.module_from_spec(spec)
sys.modules["run_baseline"] = run_baseline
spec.loader.exec_module(run_baseline)


# --- denoising step count -------------------------------------------------------


@pytest.mark.parametrize("steps,strength,expected", [
    (50, 0.8, 40),   # the configured baseline; the T4 progress bar showed exactly this
    (50, 0.5, 25),
    (50, 0.6, 30),
    (50, 0.7, 35),
    (50, 1.0, 50),
])
def test_denoising_steps_matches_pipeline_truncation(steps, strength, expected):
    assert run_baseline.denoising_steps(steps, strength) == expected


def test_denoising_steps_never_returns_zero():
    """A near-zero strength must still run one step, not divide by zero later."""
    assert run_baseline.denoising_steps(50, 0.001) == 1


def test_denoising_steps_clamps_above_requested_steps():
    """get_timesteps() takes a min against num_inference_steps; mirror it."""
    assert run_baseline.denoising_steps(50, 1.4) == 50


# --- sweep spec parsing ---------------------------------------------------------


class FakeArgs:
    def __init__(self, seed=None, steps=None, strength=None, guidance=None):
        self.seed = seed
        self.steps = steps
        self.strength = strength
        self.guidance = guidance
        self.sweep = None


CFG = {"seed": 20261103, "num_inference_steps": 50, "strength": 0.8,
       "guidance_scale": 6.0}


def test_empty_spec_falls_back_to_config():
    got = run_baseline.parse_sweep_spec("", CFG, FakeArgs())
    assert got == {"seed": 20261103, "steps": 50, "strength": 0.8, "guidance": 6.0}


def test_spec_overrides_only_the_keys_it_names():
    got = run_baseline.parse_sweep_spec("s=0.6,g=3.5", CFG, FakeArgs())
    assert got["strength"] == 0.6
    assert got["guidance"] == 3.5
    assert got["steps"] == 50          # untouched
    assert got["seed"] == 20261103     # untouched


def test_long_key_names_and_whitespace_are_accepted():
    got = run_baseline.parse_sweep_spec(" strength=0.5 , guidance = 4 , n=40 ",
                                        CFG, FakeArgs())
    assert (got["strength"], got["guidance"], got["steps"]) == (0.5, 4.0, 40)


def test_command_line_flag_beats_config_but_loses_to_the_spec():
    args = FakeArgs(strength=0.7, guidance=5.0)
    got = run_baseline.parse_sweep_spec("s=0.6", CFG, args)
    assert got["strength"] == 0.6   # the spec wins
    assert got["guidance"] == 5.0   # the flag wins over the config's 6.0


def test_seed_and_steps_stay_integers():
    got = run_baseline.parse_sweep_spec("seed=7,n=30", CFG, FakeArgs())
    assert isinstance(got["seed"], int)
    assert isinstance(got["steps"], int)


@pytest.mark.parametrize("bad", ["0.6", "x=0.6", "s"])
def test_malformed_spec_raises_valueerror(bad):
    with pytest.raises(ValueError):
        run_baseline.parse_sweep_spec(bad, CFG, FakeArgs())


# --- run list -------------------------------------------------------------------


def test_run_specs_without_sweep_yields_exactly_one_run():
    args = FakeArgs()
    assert len(run_baseline.run_specs(args, CFG)) == 1


def test_run_specs_preserves_sweep_order():
    args = FakeArgs()
    args.sweep = ["s=0.5", "s=0.6", "s=0.7", "s=0.6,g=3.5"]
    got = run_baseline.run_specs(args, CFG)
    assert [r["strength"] for r in got] == [0.5, 0.6, 0.7, 0.6]
    assert [r["guidance"] for r in got] == [6.0, 6.0, 6.0, 3.5]


# --- VAE ceiling ----------------------------------------------------------------


def test_vae_ceiling_reads_the_stage3_manifest_or_returns_none():
    """Must never raise: the ceiling is a reporting aid, not a precondition."""
    got = run_baseline.vae_ceiling_psnr()
    assert got is None or isinstance(got, float)
