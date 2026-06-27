# Figure reproduction — verification

The five figures in this release are **regenerated from the released data artifacts by clean code**
(`cot_steering/figures.py`, driven by `generate_figures.py` / the notebook) — never by copying the
reference image/PDF files. This document records the numeric + visual comparison against the
reference figures from the blog post *"A 2,880-number steering vector gives a reasoning model the
chain-of-thought control that fine-tuning does."*

Reproduce + check the numbers yourself:

```bash
python generate_figures.py --verify          # auto-loads figure_data (HF, local fallback), asserts the numbers
python generate_figures.py --source hf --verify     # specifically exercise the Hugging Face load path
python generate_figures.py --source local --verify  # fully offline, committed figure_data/
python tests/test_release.py                 # the full test suite (verify both sources, parity, consistency)
```

`--verify` runs **31 checks**: 22 plotted-value assertions (table below) plus 9 structural/relative
checks that guard the headline *messages* (the paired vector−fine-tune CI brackets 0; both uplift CIs
clear +10pp; the format specifier is the most-attended instruction part for both bullet & numbered;
the "your reasoning" reference does not rise under steering).

## Figure → data map (each figure's exact source + plotted quantities)

| figure | source artifact(s) | plotted quantity |
|---|---|---|
| **fig1_headline** | `steer_deliverable_gL10.json`, `ft_deliverable_cdel_vs_ctrldel.json` | (L) per-held-out-instruction strict CoT-control compliance (`effective_control`) for base / fine-tune / steering vector, n=100 each, 9 held-out instructions; (R) aggregate uplift over base for FT and the vector with 95% cluster-bootstrap CIs, the paired vector−FT difference, and the +10pp dashed line |
| **fig2_gradient_unlocks_formatting** | `steer_deliverable_gL10.json` (bullet base/vector/FT), `steer_eval_heldout_analysis.json` (average-difference direction, n=39), `fig2_random_null.json` (random vectors, derived) | held-out bullet `effective_control` for base / average-difference direction / random vector / gradient-trained vector / fine-tune, Wilson intervals |
| **fig3_mechanism** | `mech_qkov.json` | fraction of the formatting effect reproduced by patching the steered attention **pattern** vs the steered attention **values**, for bullet & numbered; error bars span the per-task min–max range (the `lo`/`hi` fields, = min/max over the 12 contexts) |
| **fig4_attention_tokens** | `fig4_token_shading.json` (precomputed from `tok_subspan.json` + the o200k_base token layout) | the bullet & numbered instruction text, each token shaded by the per-part average attention increase (steered − base) |
| **fig5_subspan** | `fig5_subspan_attention.json` (derived from `tok_subspan_attn.npz` per-example tensors) | attention onto each instruction part (format-specifier / "your reasoning" / directive verbs / other), base vs steered, bullet & numbered, pooled over recruited late heads, bootstrap CIs |

Three figure-data files are not verbatim summaries; they are **derived from the raw run artifacts on
CPU at ~$0** by `precompute_figure_data.py` (no model generation): the fig5 base-vs-steered per-part
attention CIs (from the per-example `tok_subspan_attn.npz`); the fig4 per-token shading layout (the
o200k_base tokenization of the two instructions + each part's per-token-average delta from
`tok_subspan.json`); and the fig2 random-vector bullet bar (`effective_control` of five random
matched-norm vectors, pooled 0/500 → 0.0%, recomputed from the raw judged generations with the
project's strict `effective_control` definition). Doing the fig4 tokenization at precompute time keeps
the plot path **offline** (no tokenizer download); `tiktoken` is only needed to re-run precompute.

## Numeric verification (`generate_figures.py --verify`, all PASS)

| figure | quantity | regenerated | reference |
|---|---|--:|--:|
| fig1 | bullet: base / FT / vector | 0 / 52 / 48 % | 0 / 52 / 48 % |
| fig1 | aggregate uplift: FT | +12.3pp [+10.4,+14.2] | +12.3pp [+10.4,+14.2] |
| fig1 | aggregate uplift: steering vector | +12.8pp [+10.7,+14.9] | +12.8pp [+10.7,+14.9] |
| fig1 | paired vector − FT difference | +0.4pp [−1.9,+2.8] | +0.4pp [−1.9,+2.8] |
| fig2 | bullet: base / avg-diff(n=39) / random / vector / FT | 0 / 0 / 0 / 48 / 52 % | 0 / 0 / 0 / 48 / 52 % |
| fig3 | bullet: pattern / value | 71 / 20 % | 71 / 20 % |
| fig3 | numbered: pattern / value | 62 / 16 % | 62 / 16 % |
| fig4 | bullet per-part Δattn: spec > "your reasoning" | spec 0.39 ≫ cot −0.05 | specifier darkest |
| fig5 | bullet format-specifier attention: base → steered | 2.29 → 6.52 | 2.3 → 6.5 |
| fig5 | numbered format-specifier attention: base → steered | 3.95 → 8.59 | 3.9 → 8.6 |
| fig5 | total instruction attention: bullet / numbered | 12.3→18.4 / 12.8→18.5 | 12.3→18.4 / 12.8→18.5 |

All 21 asserted key numbers match the published reference (see `generate_figures.py --verify`); the
remaining table rows above are read directly from the released summaries and are consistent with them.

## Visual comparison

The regenerated `fig1`–`fig5` were eyeballed side-by-side against the reference PNGs: same panels,
bars, ordering, axes, tick labels, legends, titles, error bars, and the same qualitative message.
Differences are **cosmetic only** (exact font/DPI/whitespace), as expected for a clean re-plot. The
fig3 reference is the single "pattern vs value" panel (the published fig3); fig4/fig5 use the
Segment-15 token-level attention artifacts (the published per-token + per-part attention figures).

## Honesty notes

- fig4: per-token attention was logged only for the bullet instruction at the format onset; both the
  bullet and numbered instructions are shaded by their instruction part's per-token-average increase
  (parts located by substring matching), exactly as in the published figure. This is the one figure
  whose shading is a part-average rather than a literal per-token value for every token — stated in
  the figure caption and the blog post. The token→part assignment is precomputed into
  `fig4_token_shading.json` (the two instruction strings are pure ASCII, so the o200k_base
  token→char alignment is exact).
- fig3/fig5 error bars: fig3 brackets are the **per-task min–max range** over the 12 contexts (the
  `lo`/`hi` fields of `mech_qkov.json`), matching the reference caption; fig5 bars carry 95% bootstrap
  CIs over examples (deterministic, seed 0).
- The fig5 "total instruction attention 12.3→18.4 / 12.8→18.5" row is read directly from
  `tok_subspan.json` (`instruction` span base/steer); it is consistent with the plotted per-part bars
  but is not one of the 21 asserted `--verify` checks (which cover the format-specifier sub-span).
- No reproduction gap: every plotted quantity is regenerated from the released artifacts and matches
  the reference numerically and visually.
