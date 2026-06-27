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
REFERENCE = {
    "fig1_headline": [
        ("bullet vector %", lambda k: k["bullet"]["vec"], 48.0, 0.5),
        ("bullet FT %", lambda k: k["bullet"]["ft"], 52.0, 0.5),
        ("bullet base %", lambda k: k["bullet"]["base"], 0.0, 0.5),
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
        # the format specifier darkens most; the "your reasoning" reference stays pale.
        ("bullet spec > cot_target", lambda k: k["bullet"]["spec"] - k["bullet"]["cot_target"], 0.0,
         None),  # checked as >0 below
    ],
    "fig5_subspan": [
        ("bullet spec base", lambda k: k["bullet"]["spec"]["base"], 2.3, 0.15),
        ("bullet spec steer", lambda k: k["bullet"]["spec"]["steer"], 6.5, 0.15),
        ("numbered spec base", lambda k: k["numbered"]["spec"]["base"], 3.9, 0.15),
        ("numbered spec steer", lambda k: k["numbered"]["spec"]["steer"], 8.6, 0.15),
    ],
}


def _verify(all_keys: dict) -> bool:
    ok = True
    for name, checks in REFERENCE.items():
        if name not in all_keys:
            continue
        keys = all_keys[name]
        for label, getter, expected, tol in checks:
            got = getter(keys)
            if tol is None:  # special-case: require strictly greater than `expected`
                passed = got > expected
                detail = f"{got:.4f} > {expected}"
            else:
                passed = abs(got - expected) <= tol
                detail = f"{got:.4f} vs {expected} (tol {tol})"
            mark = "OK " if passed else "XX "
            print(f"  [{mark}] {name}: {label}: {detail}")
            ok = ok and passed
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "hf", "local"], default="auto",
                    help="where to load figure-summary artifacts from (default: auto = HF then local)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory for the figures")
    ap.add_argument("--only", nargs="+", default=None,
                    help="regenerate only these figures (names from cot_steering.figures.FIGURES)")
    ap.add_argument("--verify", action="store_true",
                    help="assert the key plotted numbers match the published reference values")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    names = args.only or list(figures.FIGURES)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_keys = {}
    import matplotlib.pyplot as plt
    for name in names:
        if name not in figures.FIGURES:
            ap.error(f"unknown figure '{name}'. Choices: {list(figures.FIGURES)}")
        fig, keys = figures.FIGURES[name](source=args.source)
        for ext in args.formats:
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        all_keys[name] = keys
        print(f"[figures] saved {name} -> {out_dir}")

    (out_dir / "figure_key_numbers.json").write_text(json.dumps(all_keys, indent=2))
    print(f"[figures] wrote {out_dir / 'figure_key_numbers.json'}")

    if args.verify:
        print("\n[verify] checking key plotted numbers against the published reference:")
        if _verify(all_keys):
            print("[verify] ALL CHECKS PASSED")
        else:
            print("[verify] MISMATCH -- see XX lines above")
            sys.exit(1)


if __name__ == "__main__":
    main()
