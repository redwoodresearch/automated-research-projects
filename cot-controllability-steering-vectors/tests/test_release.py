#!/usr/bin/env python3
"""Tests for the CoT-controllability release package.

Runs as a plain script (``python tests/test_release.py``) or under pytest
(``pytest tests/``). All tests are CPU-only and require no model generation. Network-dependent
tests (the Hugging Face load path) skip cleanly when HF is unreachable.

Covered:
  * verify the regenerated figures' key numbers from the **local** committed figure_data;
  * verify them from **Hugging Face** (skips offline) -- exercises the HF path *without* the
    silent local fallback;
  * **local == HF parity** for every figure_data file (skips offline);
  * **cross-artifact consistency**: the derived fig4/fig5 summaries agree with the raw
    ``tok_subspan.json`` (so a precompute regression is caught even before --verify);
  * the figures actually write non-empty PNG/PDF files;
  * the **fully-offline** path (``HF_HUB_OFFLINE=1`` + ``--source local``) succeeds.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

import generate_figures as gf  # noqa: E402
from cot_steering import artifacts, figures  # noqa: E402


class SkipTest(Exception):
    """Raised to skip a test when an external dependency (HF network) is unavailable."""


def _all_keys(source):
    return {name: fn(source=source)[1] for name, fn in figures.FIGURES.items()}


# --------------------------------------------------------------------------------------------
def test_verify_local():
    """The 21 numeric checks + structural predicates pass from the committed figure_data."""
    assert gf.verify(_all_keys("local")), "verify() failed on local figure_data"


def test_verify_hf():
    """Same checks pass when figure_data is loaded from Hugging Face (no silent local fallback)."""
    try:
        artifacts.figure_data_path("steer_deliverable_gL10.json", source="hf")
    except Exception as exc:  # network / auth / repo issue
        raise SkipTest(f"Hugging Face unreachable: {exc}")
    assert gf.verify(_all_keys("hf")), "verify() failed on Hugging Face figure_data"


def test_local_hf_parity():
    """Every figure_data file is byte-equal (numerically) between the local copy and HF."""
    files = artifacts.RAW_SUMMARY_FILES + [
        "fig2_random_null.json", "fig4_token_shading.json", "fig5_subspan_attention.json"]
    try:
        for fn in files:
            local = json.loads(artifacts.figure_data_path(fn, source="local").read_text())
            hf = json.loads(artifacts.figure_data_path(fn, source="hf").read_text())
            assert _deep_close(local, hf), f"local != HF for {fn}"
    except SkipTest:
        raise
    except Exception as exc:
        if "Hugging Face" in str(exc) or "huggingface" in str(exc).lower() or "Connection" in str(exc):
            raise SkipTest(f"Hugging Face unreachable: {exc}")
        raise


def test_cross_artifact_consistency():
    """fig4/fig5 derived summaries agree with the raw tok_subspan.json (pure local, no network)."""
    tok = json.loads(artifacts.figure_data_path("tok_subspan.json", source="local").read_text())["attn_mass"]
    fig5 = json.loads(artifacts.figure_data_path("fig5_subspan_attention.json", source="local").read_text())
    fig4 = json.loads(artifacts.figure_data_path("fig4_token_shading.json", source="local").read_text())
    spans = ["spec", "cot_target", "directive", "rest"]
    for cond in ("bullet", "numbered"):
        # fig5 means == tok_subspan base/steer per span
        for s in spans:
            assert math.isclose(fig5[cond]["spans"][s]["base"]["mean"], tok[cond][s]["base"], rel_tol=1e-6), \
                f"fig5 {cond}/{s} base mean != tok_subspan"
            assert math.isclose(fig5[cond]["spans"][s]["steer"]["mean"], tok[cond][s]["steer"], rel_tol=1e-6), \
                f"fig5 {cond}/{s} steer mean != tok_subspan"
        # the four sub-spans partition the instruction -> they sum to the instruction total
        for arm in ("base", "steer"):
            total = sum(tok[cond][s][arm] for s in spans)
            assert math.isclose(total, tok[cond]["instruction"][arm], rel_tol=1e-5), \
                f"{cond} {arm}: sub-spans sum {total} != instruction {tok[cond]['instruction'][arm]}"
        # fig4 per-part value == part delta / (token count for that part)
        for s in spans:
            n = sum(1 for tk in fig4[cond] if math.isclose(tk["v"], fig4["part_values"][cond][s]))
            assert n >= 1, f"fig4 {cond}/{s} part value not present on any token"
            expect = tok[cond][s]["delta"] / n
            assert math.isclose(fig4["part_values"][cond][s], expect, rel_tol=1e-6), \
                f"fig4 {cond}/{s} part value {fig4['part_values'][cond][s]} != delta/count {expect}"


def test_precompute_roundtrip():
    """Re-derive fig4/fig5 from raw artifacts and assert they match the committed figure_data.

    Opt-in: set ``COT_RAW_DIR=/path/to/results`` (a dir with tok_subspan.json + tok_subspan_attn.npz
    + meta). Skips otherwise so the default suite stays fast and network-free.
    """
    raw = os.environ.get("COT_RAW_DIR")
    if not raw:
        raise SkipTest("set COT_RAW_DIR to a raw results dir to run the precompute round-trip")
    import precompute_figure_data as pre
    raw_dir = Path(raw)
    fig5_new = pre.derive_fig5(raw_dir)
    fig4_new = pre.derive_fig4(raw_dir)
    fig5_old = json.loads(artifacts.figure_data_path("fig5_subspan_attention.json", source="local").read_text())
    fig4_old = json.loads(artifacts.figure_data_path("fig4_token_shading.json", source="local").read_text())
    assert _deep_close(fig5_new, fig5_old), "re-derived fig5 != committed"
    assert _deep_close(fig4_new, fig4_old), "re-derived fig4 != committed"


def test_outputs_written():
    """generate_all writes non-empty png + pdf for every figure."""
    with tempfile.TemporaryDirectory() as d:
        figures.generate_all(d, source="local", verbose=False)
        for name in figures.FIGURES:
            for ext in ("png", "pdf"):
                p = Path(d) / f"{name}.{ext}"
                assert p.exists() and p.stat().st_size > 1000, f"missing/empty {p}"


def test_offline_local():
    """With HF disabled, the local source still regenerates + verifies (no network, no tokenizer)."""
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        assert gf.verify(_all_keys("local")), "offline local verify failed"
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


# --------------------------------------------------------------------------------------------
def _deep_close(a, b, rel_tol=1e-9, abs_tol=1e-9):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_deep_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_deep_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except SkipTest as s:
            print(f"SKIP  {t.__name__}: {s}")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
