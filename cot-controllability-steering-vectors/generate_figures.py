#!/usr/bin/env python3
"""Regenerate the five publication figures (fig1-fig5) from the released artifacts.

CPU only, seconds to run, **no model generation / GPU / training**. By default the small
figure-summary JSONs are loaded from Hugging Face (org ``automated-alignment-science``)
with a fallback to the copies committed under ``figure_data/``.

Usage::

    python generate_figures.py                 # -> figures/fig1..fig5 (.png + .pdf)
    python generate_figures.py --source local  # force the committed local summaries
    python generate_figures.py --verify        # also assert the key numbers reproduce
    python generate_figures.py --only fig1_headline fig5_subspan

The ``--verify`` flag checks the load-bearing quantities against the published reference
values (see ``figures/REPRODUCTION.md``) and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cot_steering import figures

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "figures"

# Reference values the regenerated figures must match (from the published blog post +
# the verified result artifacts). (value, abs_tolerance_in_the_plotted_unit).
# Numeric checks: (label, getter, expected, abs_tolerance). Cover the load-bearing plotted values.
REFERENCE = {
    "fig1_headline": [
        ("bullet base/FT/vector %", lambda k: k["bullet"]["base"], 0.0, 0.5),
        ("bullet FT %", lambda k: k["bullet"]["ft"], 52.0, 0.5),
        ("bullet vector %", lambda k: k["bullet"]["vec"], 48.0, 0.5),
        ("terse base %", lambda k: k["terse"]["base"], 14.0, 0.5),
        ("terse FT %", lambda k: k["terse"]["ft"], 61.0, 0.5),
        ("terse vector %", lambda k: k["terse"]["vec"], 69.0, 0.5),
        ("numbered FT %", lambda k: k["numbered"]["ft"], 9.0, 0.5),
        ("numbered vector %", lambda k: k["numbered"]["vec"], 8.0, 0.5),
        ("vector uplift pp", lambda k: k["vec_uplift_pp"], 12.8, 0.2),
        ("FT uplift pp", lambda k: k["ft_uplift_pp"], 12.3, 0.2),
        ("paired diff pp", lambda k: k["paired_diff_pp"], 0.4, 0.2),
    ],
    "fig2_gradient_unlocks_formatting": [
        ("base model bullet %", lambda k: k["base model"]["pct"], 0.0, 0.5),
        ("avg-diff bullet %", lambda k: k["average-difference direction"]["pct"], 0.0, 0.5),
        ("avg-diff n", lambda k: k["average-difference direction"]["n"], 39, 0),
        ("random bullet %", lambda k: k["random vector (same size)"]["pct"], 0.0, 0.5),
        ("gradient bullet %", lambda k: k["gradient-trained steering vector"]["pct"], 48.0, 0.5),
        ("FT bullet %", lambda k: k["fine-tuned (LoRA)"]["pct"], 52.0, 0.5),
    ],
    "fig3_mechanism": [
        ("bullet pattern %", lambda k: k["bullet"]["pattern_pct"], 71.0, 1.5),
        ("bullet value %", lambda k: k["bullet"]["value_pct"], 20.0, 1.5),
        ("numbered pattern %", lambda k: k["numbered"]["pattern_pct"], 62.0, 1.5),
        ("numbered value %", lambda k: k["numbered"]["value_pct"], 16.0, 1.5),
    ],
    "fig4_attention_tokens": [
        ("bullet spec increase", lambda k: k["bullet"]["spec"], 0.385, 0.03),
        ("numbered spec increase", lambda k: k["numbered"]["spec"], 0.211, 0.03),
    ],
    "fig5_subspan": [
        ("bullet spec base", lambda k: k["bullet"]["spec"]["base"], 2.3, 0.15),
        ("bullet spec steer", lambda k: k["bullet"]["spec"]["steer"], 6.5, 0.15),
        ("numbered spec base", lambda k: k["numbered"]["spec"]["base"], 3.9, 0.15),
        ("numbered spec steer", lambda k: k["numbered"]["spec"]["steer"], 8.6, 0.15),
        ("bullet cot_target base", lambda k: k["bullet"]["cot_target"]["base"], 4.5, 0.2),
        ("bullet cot_target steer", lambda k: k["bullet"]["cot_target"]["steer"], 4.0, 0.2),
    ],
}

# Structural checks: (label, predicate(keys) -> bool, detail string). Guard the *relative* claims
# (the headline messages), not just point values.
PREDICATES = {
    "fig1_headline": [
        ("paired vector-FT diff CI brackets 0",
         lambda k: k["paired_diff_ci"][0] < 0 < k["paired_diff_ci"][1],
         lambda k: f"CI {[round(x, 1) for x in k['paired_diff_ci']]}"),
        ("vector uplift CI clears +10pp", lambda k: k["vec_uplift_ci"][0] > 10.0,
         lambda k: f"ci_lo {k['vec_uplift_ci'][0]:.1f} > 10"),
        ("FT uplift CI clears +10pp", lambda k: k["ft_uplift_ci"][0] > 10.0,
         lambda k: f"ci_lo {k['ft_uplift_ci'][0]:.1f} > 10"),
    ],
    "fig4_attention_tokens": [
        ("bullet: specifier is the max part",
         lambda k: k["bullet"]["spec"] == max(k["bullet"].values()),
         lambda k: "spec %.3f = max(%s)" % (k["bullet"]["spec"],
                   {p: round(v, 3) for p, v in k["bullet"].items()})),
        ("numbered: specifier is the max part",
         lambda k: k["numbered"]["spec"] == max(k["numbered"].values()),
         lambda k: f"spec {k['numbered']['spec']:.3f}"),
        ("bullet: 'your reasoning' barely moves (< specifier)",
         lambda k: k["bullet"]["cot_target"] < k["bullet"]["spec"] * 0.2,
         lambda k: f"cot_target {k['bullet']['cot_target']:.3f} << spec {k['bullet']['spec']:.3f}"),
    ],
    "fig5_subspan": [
        ("bullet: specifier roughly triples (steer >> base)",
         lambda k: k["bullet"]["spec"]["steer"] > 2.5 * k["bullet"]["spec"]["base"],
         lambda k: f"{k['bullet']['spec']['base']:.2f} -> {k['bullet']['spec']['steer']:.2f}"),
        ("bullet: 'your reasoning' does NOT rise",
         lambda k: k["bullet"]["cot_target"]["steer"] <= k["bullet"]["cot_target"]["base"],
         lambda k: f"{k['bullet']['cot_target']['base']:.2f} -> {k['bullet']['cot_target']['steer']:.2f}"),
        ("numbered: specifier roughly doubles (steer >> base)",
         lambda k: k["numbered"]["spec"]["steer"] > 1.8 * k["numbered"]["spec"]["base"],
         lambda k: f"{k['numbered']['spec']['base']:.2f} -> {k['numbered']['spec']['steer']:.2f}"),
    ],
}


def verify(all_keys: dict) -> bool:
    """Assert each regenerated figure's key numbers + relative claims match the published reference."""
    ok = True
    for name, checks in REFERENCE.items():
        if name not in all_keys:
            continue
        keys = all_keys[name]
        for label, getter, expected, tol in checks:
            got = getter(keys)
            passed = abs(got - expected) <= tol
            mark = "OK " if passed else "XX "
            print(f"  [{mark}] {name}: {label}: {got:.4f} vs {expected} (tol {tol})")
            ok = ok and passed
    for name, preds in PREDICATES.items():
        if name not in all_keys:
            continue
        keys = all_keys[name]
        for label, pred, detail in preds:
            passed = bool(pred(keys))
            mark = "OK " if passed else "XX "
            print(f"  [{mark}] {name}: {label}: {detail(keys)}")
            ok = ok and passed
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "hf", "local"], default=None,
                    help="where to load figure-summary artifacts from (default: the "
                         "COT_ARTIFACT_SOURCE env var, else auto = HF then local fallback)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory for the figures")
    ap.add_argument("--only", nargs="+", default=None,
                    help="regenerate only these figures (names from cot_steering.figures.FIGURES)")
    ap.add_argument("--verify", action="store_true",
                    help="assert the key plotted numbers match the published reference values")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    names = args.only or list(figures.FIGURES)
    unknown = [n for n in names if n not in figures.FIGURES]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}. Choices: {list(figures.FIGURES)}")
    out_dir = Path(args.out)
    all_keys = figures.generate_all(out_dir, source=args.source, names=names, formats=args.formats)

    (out_dir / "figure_key_numbers.json").write_text(json.dumps(all_keys, indent=2))
    print(f"[figures] wrote {out_dir / 'figure_key_numbers.json'}")

    if args.verify:
        print("\n[verify] checking key plotted numbers against the published reference:")
        if verify(all_keys):
            print("[verify] ALL CHECKS PASSED")
        else:
            print("[verify] MISMATCH -- see XX lines above")
            sys.exit(1)


if __name__ == "__main__":
    main()
