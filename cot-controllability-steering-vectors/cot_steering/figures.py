"""Regenerate the five publication figures from the released summary artifacts.

Each ``figN_*`` function loads only small JSON summaries (via :mod:`cot_steering.artifacts`,
which fetches from Hugging Face with a local fallback), builds the matplotlib figure, and
returns ``(fig, key_numbers)``. ``key_numbers`` is a small dict of the load-bearing values
plotted, so callers can *assert* the figure reproduces the reference numerically.

These figures plot saved summary metrics only -- there is **no model generation, no GPU,
no training**. See ``generate_figures.py`` for the CPU entry point and ``figures/REPRODUCTION.md``
for the numeric/visual comparison against the reference figures.

Figure -> data map
------------------
* fig1  steer_deliverable_gL10.json + ft_deliverable_cdel_vs_ctrldel.json
* fig2  steer_deliverable_gL10.json (bullet base/vector/FT) + steer_eval_heldout_analysis.json
        (average-difference direction, n=39) + fig2_random_null.json (random vectors)
* fig3  mech_qkov.json  (attention-pattern vs attention-value patch fractions)
* fig4  tok_subspan.json (per-instruction-part attention deltas; tokens shaded by part avg)
* fig5  fig5_subspan_attention.json (base vs steered attention per part, bootstrap CIs)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps, colors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .artifacts import load_figure_json  # noqa: E402

# ---- shared styling ---------------------------------------------------------------------
C_BASE = "#9aa0a6"   # grey  - base model
C_FT = "#1f77b4"     # blue  - fine-tuned (LoRA)
C_VEC = "#d62728"    # red   - steering vector
C_DOM = "#c8b9a6"    # tan   - average-difference direction
C_NULL = "#dcdcdc"   # light - random vector
C_PATT = "#6a51a3"   # purple - attention pattern
C_VAL = "#c7c0db"    # light purple - attention values

_RC = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 130,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
}

# Held-out instruction display order + labels (fig1 left panel).
FIG1_ORDER = ["bullet", "terse_25w", "numbered", "xml_steps", "no_word_so",
              "initial_caps", "include_exactly_twice", "section_headers", "child_explanation"]
FIG1_LABELS = {
    "bullet": "bullet lines", "terse_25w": "terse (\u226425 words)", "numbered": "numbered lines",
    "xml_steps": "XML step tags", "no_word_so": 'no word "so"', "initial_caps": "Capitalize Each Word",
    "include_exactly_twice": "include a word \u00d72", "section_headers": "section headers",
    "child_explanation": "explain to a child",
}


def _wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# =========================================================================================
# FIG 1 -- a frozen-weights steering vector reproduces fine-tuning's held-out CoT control
# =========================================================================================
def fig1_headline(source=None):
    sd = load_figure_json("steer_deliverable_gL10.json", source=source)
    ftd = load_figure_json("ft_deliverable_cdel_vs_ctrldel.json", source=source)
    pi = sd["per_instruction"]
    base = [pi[k]["base"]["effective_control"] * 100 for k in FIG1_ORDER]
    ft = [pi[k]["ft"]["effective_control"] * 100 for k in FIG1_ORDER]
    vec = [pi[k]["gL10"]["effective_control"] * 100 for k in FIG1_ORDER]

    with plt.rc_context(_RC):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                       gridspec_kw={"width_ratios": [3, 1.25]})
        x = np.arange(len(FIG1_ORDER)); w = 0.27
        axA.bar(x - w, base, w, label="base model", color=C_BASE)
        axA.bar(x, ft, w, label="fine-tuned (LoRA)", color=C_FT)
        axA.bar(x + w, vec, w, label="steering vector", color=C_VEC)
        axA.set_xticks(x)
        axA.set_xticklabels([FIG1_LABELS[k] for k in FIG1_ORDER], rotation=40, ha="right")
        axA.set_ylabel("strict CoT-control compliance (%)")
        axA.set_title("Per held-out instruction (n=100 tasks each)")
        axA.set_ylim(0, 76)
        axA.legend(loc="upper right", ncol=1)
        axA.annotate("formatting category\n(never trained on)", xy=(0.5, 60), xytext=(2.6, 64),
                     ha="left", fontsize=9, color="#444")

        m = sd["macros"]["all_heldout"]
        ftup = ftd["macros"]["all_heldout"]["uplift"]
        gup = m["gL10_uplift_vs_base"]
        diff = m["gL10_minus_ft"]
        items = [
            ("fine-tuned\n(uplift vs base)", ftup["point"] * 100,
             (ftup["point"] - ftup["ci_lo"]) * 100, (ftup["ci_hi"] - ftup["point"]) * 100, C_FT),
            ("steering vector\n(uplift vs base)", gup["point"] * 100,
             (gup["point"] - gup["ci_lo"]) * 100, (gup["ci_hi"] - gup["point"]) * 100, C_VEC),
            ("vector \u2212 fine-tuned\n(difference)", diff["point"] * 100,
             (diff["point"] - diff["ci_lo"]) * 100, (diff["ci_hi"] - diff["point"]) * 100, "#555"),
        ]
        yp = np.arange(len(items))[::-1]
        for y, (lab, pt, lo, hi, c) in zip(yp, items):
            axB.errorbar(pt, y, xerr=[[lo], [hi]], fmt="o", color=c, capsize=4, ms=7)
            axB.text(pt, y - 0.30, f"{pt:+.1f}", ha="center", fontsize=9, color=c)
        axB.axvline(0, color="#bbb", lw=1, zorder=0)
        axB.axvline(10, color="#2ca02c", lw=1, ls="--", zorder=0)
        axB.text(10, -1.08, "+10pp\nmin. effect", color="#2ca02c", fontsize=8, ha="center")
        axB.set_yticks(yp); axB.set_yticklabels([i[0] for i in items], fontsize=9)
        axB.set_xlabel("percentage points")
        axB.set_title("Aggregate over 9 held-out\ninstructions (95% CI)")
        axB.set_xlim(-6, 18); axB.set_ylim(-1.35, 2.55)
        fig.suptitle(
            "A frozen-weights steering vector reproduces fine-tuning's held-out CoT control",
            fontsize=13, y=1.02)

    keys = {
        "bullet": {"base": base[0], "ft": ft[0], "vec": vec[0]},
        "terse": {"base": base[1], "ft": ft[1], "vec": vec[1]},
        "numbered": {"base": base[2], "ft": ft[2], "vec": vec[2]},
        "ft_uplift_pp": ftup["point"] * 100,
        "ft_uplift_ci": [ftup["ci_lo"] * 100, ftup["ci_hi"] * 100],
        "vec_uplift_pp": gup["point"] * 100,
        "vec_uplift_ci": [gup["ci_lo"] * 100, gup["ci_hi"] * 100],
        "paired_diff_pp": diff["point"] * 100,
        "paired_diff_ci": [diff["ci_lo"] * 100, diff["ci_hi"] * 100],
    }
    return fig, keys


# =========================================================================================
# FIG 2 -- held-out bullet: only the gradient objective (or fine-tuning) unlocks formatting
# =========================================================================================
def fig2_gradient_unlocks_formatting(source=None):
    sd = load_figure_json("steer_deliverable_gL10.json", source=source)
    se = load_figure_json("steer_eval_heldout_analysis.json", source=source)
    rn = load_figure_json("fig2_random_null.json", source=source)
    b = sd["per_instruction"]["bullet"]
    avg_diff = se["per_instruction"]["bullet"]["real"]  # single-layer diff-of-means direction (n=39)

    arms = [
        ("base model", b["base"]["effective_control"] * 100, b["base"]["n"], C_BASE),
        ("average-difference\ndirection", avg_diff["effective_control"] * 100, avg_diff["n"], C_DOM),
        ("random vector\n(same size)", rn["p"] * 100, rn["n"], C_NULL),
        ("gradient-trained\nsteering vector", b["gL10"]["effective_control"] * 100, b["gL10"]["n"], C_VEC),
        ("fine-tuned\n(LoRA)", b["ft"]["effective_control"] * 100, b["ft"]["n"], C_FT),
    ]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for i, (lab, p, n, c) in enumerate(arms):
            lo, hi = _wilson(p / 100, n)
            ax.bar(i, p, 0.62, color=c, edgecolor="#666", lw=0.5)
            ax.errorbar(i, p, yerr=[[(p / 100 - lo) * 100], [(hi - p / 100) * 100]], fmt="none",
                        ecolor="#333", capsize=4, lw=1)
            ax.text(i, p + (3 if p > 1 else 2), f"{p:.0f}%", ha="center", fontsize=10)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=9.5)
        ax.set_ylabel("bullet-format CoT compliance (%)")
        ax.set_ylim(0, 62)
        ax.set_title('Held-out "bullet" formatting: same data, only the fitting procedure differs')

    keys = {lab.replace("\n", " "): {"pct": round(p, 1), "n": n} for lab, p, n, _ in arms}
    return fig, keys


# =========================================================================================
# FIG 3 -- the effect flows through where the heads look, not what they read
# =========================================================================================
def fig3_mechanism(source=None):
    mq = load_figure_json("mech_qkov.json", source=source)["form"]
    conds = ["bullet", "numbered"]
    patt = [mq[c]["pattern_full"]["frac"] * 100 for c in conds]
    val = [mq[c]["value_full"]["frac"] * 100 for c in conds]
    patt_err = [[(mq[c]["pattern_full"]["frac"] - mq[c]["pattern_full"]["lo"]) * 100 for c in conds],
                [(mq[c]["pattern_full"]["hi"] - mq[c]["pattern_full"]["frac"]) * 100 for c in conds]]
    val_err = [[(mq[c]["value_full"]["frac"] - mq[c]["value_full"]["lo"]) * 100 for c in conds],
               [(mq[c]["value_full"]["hi"] - mq[c]["value_full"]["frac"]) * 100 for c in conds]]
    with plt.rc_context(_RC):
        fig, ax1 = plt.subplots(figsize=(7.8, 5.1))
        fig.subplots_adjust(bottom=0.26, top=0.88)
        xc = np.arange(len(conds)); w = 0.34
        ax1.bar(xc - w / 2, patt, w, yerr=patt_err, capsize=4, color=C_PATT,
                label="attention pattern (where heads look)")
        ax1.bar(xc + w / 2, val, w, yerr=val_err, capsize=4, color=C_VAL,
                label="attention values (what is read)")
        ax1.set_xticks(xc); ax1.set_xticklabels(["bullet", "numbered"])
        ax1.set_ylabel("% of the formatting effect reproduced")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, fontsize=10)
        fig.suptitle("The effect flows through where the heads look, not what they read",
                     fontsize=12.5, y=0.96)
    keys = {c: {"pattern_pct": round(patt[i], 1), "value_pct": round(val[i], 1)}
            for i, c in enumerate(conds)}
    return fig, keys


# =========================================================================================
# FIG 4 -- instruction text shaded by the per-part attention increase (steered - base)
# =========================================================================================
# The token layout + per-token shading value is precomputed into ``fig4_token_shading.json`` by
# ``precompute_figure_data.py`` (which uses the o200k_base tokenizer once). Plotting needs no
# tokenizer and no network, so this figure is fully offline like the others.
def fig4_attention_tokens(source=None):
    data = load_figure_json("fig4_token_shading.json", source=source)
    norm = colors.Normalize(vmin=0.0, vmax=data["vmax"])
    cmap = colormaps["Reds"]

    CW, LH, WRAP = 1.0, 2.0, 70
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        y = 0.0
        for c in ("bullet", "numbered"):
            ax.text(0, y + 1.0, f"{c} instruction", fontsize=11, fontweight="bold", va="bottom")
            y -= LH * 0.7
            x = 0.0
            for tok in data[c]:
                t, val = tok["t"], max(0.0, tok["v"])
                w = len(t) * CW
                if x + w > WRAP:
                    x = 0.0; y -= LH
                fc = cmap(norm(val))
                ax.add_patch(Rectangle((x, y - 0.45), w, 1.35, fc=fc, ec="none", zorder=1))
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                ax.text(x + 0.05, y + 0.2, t.replace(" ", "\u00a0"), fontsize=11, va="center",
                        ha="left", family="DejaVu Sans Mono",
                        color="white" if lum < 0.5 else "#111111", zorder=2)
                x += w
            y -= LH * 1.7
        ax.set_xlim(-1, WRAP + 1)
        ax.set_ylim(y + LH * 0.6, 2.0)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig.add_axes([0.30, 0.07, 0.40, 0.03])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("increase in attention to the token, steered \u2212 base "
                     "(per-token average within each instruction part)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    keys = {c: {g: round(v, 4) for g, v in data["part_values"][c].items()}
            for c in ("bullet", "numbered")}
    return fig, keys


# =========================================================================================
# FIG 5 -- attention onto each instruction part, base vs steered (bullet & numbered)
# =========================================================================================
_FIG5_SPANS = ["spec", "cot_target", "directive", "rest"]
_FIG5_SPAN_LABEL = {
    "spec": "format specifier\n(\u201cbulleted\u201d, \u201c'- '\u201d)",
    "cot_target": "\u201cyour reasoning\u201d\nreference",
    "directive": "directive verbs\n(\u201cwrite\u201d, \u201cstart\u201d)",
    "rest": "other words",
}


def fig5_subspan(source=None):
    fa = load_figure_json("fig5_subspan_attention.json", source=source)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
        fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.08)
        for ax, cond in zip(axes, ["bullet", "numbered"]):
            spans_d = fa[cond]["spans"]
            xc = np.arange(len(_FIG5_SPANS)); w = 0.38
            for j, (arm, col, lab) in enumerate([("base", C_BASE, "base model"),
                                                 ("steer", C_VEC, "with steering vector")]):
                pts = [spans_d[s][arm]["mean"] for s in _FIG5_SPANS]
                los = [spans_d[s][arm]["mean"] - spans_d[s][arm]["lo"] for s in _FIG5_SPANS]
                his = [spans_d[s][arm]["hi"] - spans_d[s][arm]["mean"] for s in _FIG5_SPANS]
                ax.bar(xc + (j - 0.5) * w, pts, w, yerr=[los, his], capsize=4, color=col, label=lab)
            ax.set_xticks(xc)
            ax.set_xticklabels([_FIG5_SPAN_LABEL[s] for s in _FIG5_SPANS], fontsize=9)
            ax.set_title(f"{cond} instruction")
        axes[0].set_ylabel("attention onto the sub-span\n(recruited late heads)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=10)
        fig.suptitle("Adding the steering vector raises attention onto the instruction "
                     "\u2014 concentrated on the format specifier", fontsize=12, y=0.97)
    keys = {cond: {s: {"base": round(fa[cond]["spans"][s]["base"]["mean"], 2),
                       "steer": round(fa[cond]["spans"][s]["steer"]["mean"], 2)}
                   for s in _FIG5_SPANS} for cond in ("bullet", "numbered")}
    return fig, keys


# =========================================================================================
# Registry + driver
# =========================================================================================
FIGURES = {
    "fig1_headline": fig1_headline,
    "fig2_gradient_unlocks_formatting": fig2_gradient_unlocks_formatting,
    "fig3_mechanism": fig3_mechanism,
    "fig4_attention_tokens": fig4_attention_tokens,
    "fig5_subspan": fig5_subspan,
}


def generate_all(out_dir, source=None, names=None, formats=("png", "pdf"), verbose=True):
    """Regenerate figures ``names`` (default all) to ``out_dir``; return ``{name: key_numbers}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_keys = {}
    for name in (names or list(FIGURES)):
        if name not in FIGURES:
            raise KeyError(f"unknown figure '{name}'. Choices: {list(FIGURES)}")
        fig, keys = FIGURES[name](source=source)
        for ext in formats:
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        all_keys[name] = keys
        if verbose:
            print(f"[figures] saved {name} -> {out_dir}")
    return all_keys
