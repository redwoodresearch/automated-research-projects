"""Segment 11 (mech_interp) — P2.4: per-head attention mass onto the in-context instruction span
(+ sink + prompt + reasoning-prefix), base vs gL10-steered, on the teacher-forced contexts. The
gating hypothesis predicts gL10 RAISES attention to the instruction. SLOW (Modal); cached.

Captures ALL 24 layers at the first reasoning position (where the instruction is still within the
128-token sliding window, so both attention types can reach it) AND a LONG-CONTEXT variant (a deep
read position where the instruction is >128 tokens back, so ONLY full-attention layers + the sink
can attend to it — the decode-relevant regime).

  python run_capture_attn.py [--assert-cached]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import modal

import gpt_oss_infer as G
import mech_lib as M

RESULTS = Path("results")


def build_spans(seqs, positions, seq_meta, base_spans):
    """Augment build_contexts span_ranges with header/first_tok/self_prefix relative to read pos."""
    sr = {}
    for i, (sm, pos) in enumerate(zip(seq_meta, positions)):
        rp = pos[0]
        d = dict(base_spans.get(i, {}))
        pl = sm["prompt_len"]
        d.setdefault("prompt", (0, pl))
        d["header"] = (pl, rp + 1)          # analysis header + read pos
        d["first_tok"] = (0, 1)
        d["self_prefix"] = (pl, rp + 1)
        sr[i] = {k: (int(a), int(b)) for k, (a, b) in d.items()}
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--long-prefix", type=int, default=200,
                    help="long-context variant: prefix length so the instruction is >128 back")
    args = ap.parse_args()

    arms, vnorm = M.make_arms()
    steering = arms["gL10"]
    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        tasks0 = M.subsample_tasks("heldout", M.SIZES, "heldout")
        none_prompts = [M.H.render_prompt_tokens(M.none_user_content(t), reasoning_effort="medium")
                        for t in tasks0]
        # need a longer prefix for the long-context variant
        gp = {"max_new_tokens": max(M.BASE_MNT, args.long_prefix + 16), "temperature": 0.0, "seed": 0}
        ngens = G.generate_many(obj, none_prompts, gp, cache, tracker, batch_size=12,
                                assert_cached=args.assert_cached)
        prefix_short, prefix_long = {}, {}
        for t, g in zip(tasks0, ngens):
            comp = g["token_ids"]
            cs = M.find_content_start(comp)
            if cs is not None and cs < len(comp):
                prefix_short[t["task_id"]] = comp[cs:cs + M.N_PREFIX]
                if len(comp) - cs >= args.long_prefix:
                    prefix_long[t["task_id"]] = comp[cs:cs + args.long_prefix]

        # ---- first-position capture (all 24 layers; instruction nearby) ----
        _, seqs, positions, seq_meta, base_spans = M.build_contexts(prefix_short)
        sr = build_spans(seqs, positions, seq_meta, base_spans)
        res = G.capture_attn(obj, seqs, positions, steering, {i: sr[i] for i in range(len(seqs))},
                             cache, tracker, capture_layers=list(range(24)), micro_batch=2,
                             assert_cached=args.assert_cached)
        _save("first", res, seq_meta, sr)

        # ---- long-context capture (read DEEP in the reasoning; instruction >128 back) ----
        hdr = M.analysis_header_ids()
        lseqs, lpos, lmeta, lspans = [], [], [], {}
        tasks = M.subsample_tasks("heldout", M.SIZES, "heldout")
        for cond in ["none", "bullet", "numbered"]:
            for t in tasks:
                if t["task_id"] not in prefix_long:
                    continue
                p_ids, ispan = M.instruction_span(t, cond)
                seq = list(p_ids) + hdr + list(prefix_long[t["task_id"]])
                read = len(seq) - 2     # deep read position
                # require instruction >128 tokens before read
                if ispan is not None and (read - ispan[1]) < 128 and cond != "none":
                    pass  # still record; analysis filters by distance
                i = len(lseqs)
                lseqs.append(seq); lpos.append([read])
                lmeta.append({"cond": cond, "task_id": t["task_id"], "first_pos": read,
                              "prompt_len": len(p_ids),
                              "instr_dist": (read - ispan[1]) if ispan else None})
                d = {"prompt": (0, len(p_ids)), "self_prefix": (len(p_ids), read + 1),
                     "first_tok": (0, 1)}
                if ispan is not None:
                    d["instruction"] = ispan
                lspans[i] = {k: (int(a), int(b)) for k, (a, b) in d.items()}
        lres = G.capture_attn(obj, lseqs, lpos, steering, {i: lspans[i] for i in range(len(lseqs))},
                              cache, tracker, capture_layers=list(range(24)), micro_batch=2,
                              assert_cached=args.assert_cached)
        _save("long", lres, lmeta, lspans)

    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


def _save(tag, res, seq_meta, span_ranges):
    rows = res["results"]
    span_names = ["instruction", "prompt", "self_prefix", "first_tok", "sink"]
    n = len(rows)
    nL, nH = 24, 64
    arrs = {}
    for arm in ["base", "steer"]:
        for nm in span_names:
            arrs[f"{arm}_{nm}"] = np.full((n, nL, nH), np.nan, np.float32)
    has_instr = np.zeros(n, bool)
    for r, row in enumerate(rows):
        i = row["seq_idx"]
        has_instr[r] = "instruction" in span_ranges.get(i, {})
        for arm in ["base", "steer"]:
            if arm not in row:
                continue
            for L in range(nL):
                e = row[arm].get(str(L))
                if e is None:
                    continue
                for nm in span_names:
                    if nm in e:
                        arrs[f"{arm}_{nm}"][r, L] = e[nm]
    meta = {"conds": [m["cond"] for m in seq_meta], "has_instr": has_instr.tolist(),
            "span_names": span_names, "full_attention_layers": res["full_attention_layers"],
            "instr_dist": [m.get("instr_dist") for m in seq_meta]}
    np.savez(RESULTS / f"mech_attn_{tag}.npz", **arrs)
    json.dump(meta, open(RESULTS / f"mech_attn_{tag}_meta.json", "w"), indent=2)
    print(f"wrote results/mech_attn_{tag}.npz + meta (n={n})")


if __name__ == "__main__":
    main()
