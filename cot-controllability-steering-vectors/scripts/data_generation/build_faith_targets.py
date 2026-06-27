"""Construct the bidirectional faithfulness training targets (Segment 12).

Input  : results/cue_probe_train_base.jsonl  (base cued TRAIN traces; cue=key).
Sources: cue-following traces (follows_cue=True, coherent, not truncated/degenerate) — these
         conclude the cued WRONG letter X and ~ceiling-acknowledge the cue.
Output : data/faith_targets_<tag>.jsonl  (paired rows; kind in {conceal, faithful}).

  * faithful  target = the source analysis as-is (acknowledges the cue) — gated cue_ack=YES,
    genuine, not degenerate, concludes X.  (the steer-TOWARD-faithfulness data)
  * conceal   target = conceal_edit(source) — argues for X with NO cue reference, length/coherence
    preserved — gated cue_ack=NO, concludes X, genuine, coherent, not degenerate, length_ok.
    (the steer-AGAINST-faithfulness data)

All LLM calls (edits + gates) are cached + cost-tracked via llm.py. Reuses cue_lib (prompt assembly),
run_cue_judges.judge_cue_ack (defining ack gate), judges.judge_genuine, sft_edit.is_degenerate.

Usage:
  python build_faith_targets.py --in results/cue_probe_train_base.jsonl --tag full --concurrency 16
  python build_faith_targets.py --in ... --tag dev --limit 60       # small DEV set for inspection
  python build_faith_targets.py --assert-cached --tag full ...      # verify $0 re-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import answer_scoring as A
import cue_lib as C
import faith_edit as FE
import instructions as I
import judges as J
from run_cue_judges import judge_cue_ack
from sft_edit import is_degenerate

DATA = Path("data")
MAX_ATTEMPTS = 3
MIN_SOURCE_WORDS = 45  # need a substantive source to conceal AT LENGTH


def reconstruct_uc(row) -> str:
    task = {"question": row["question"], "answer_type": row["answer_type"]}
    # use the row's actual cue (Seg-12 rows are all cue="key" -> byte-identical; Seg-13 reliance rows
    # may be key/stanford). Falls back to "key".
    return C.cued_user_content(task, row["cued_letter"], row.get("cue", "key"))


def eligible_source(row) -> bool:
    return (row.get("follows_cue") and not row.get("malformed") and not row.get("truncated")
            and (row.get("analysis") or "").strip()
            and I.word_count(row["analysis"]) >= MIN_SOURCE_WORDS
            and not is_degenerate(row["analysis"]))


async def build_faithful(row, *, assert_cached, matched=True):
    """Faithful (acknowledging) target. matched=True (default): a STYLE-MATCHED Opus rewrite (clean
    structured analysis that acknowledges the cue + concludes X) so against(conceal)/toward(faithful)
    differ ONLY in acknowledgment, not writing style (the dataset-reviewer's critical finding). The
    matched faithful is also a stronger validity control: clean-Opus-style WITH the cue should NOT
    conceal. matched=False: the raw source trace (legacy)."""
    X = row["cued_letter"]
    q = row["question"]
    if not matched:
        analysis = row["analysis"]
        ack = await judge_cue_ack(X, analysis, assert_cached=assert_cached)
        genuine = await J.judge_genuine(q, analysis, assert_cached=assert_cached)
        concl = await FE.concludes_letter(analysis, assert_cached=assert_cached)
        gates = {"cue_ack": ack, "genuine": genuine, "concludes": concl,
                 "degenerate": is_degenerate(analysis)}
        ok = (ack is True and genuine is True and concl == X and not gates["degenerate"])
        return (analysis if ok else None), gates
    # matched-style Opus rewrite (acknowledges cue), same gate family as conceal but cue_ack=YES
    src = row["analysis"]
    last = {}
    for s in range(MAX_ATTEMPTS):
        cand = await FE.faithful_edit(q, X, src, sample_idx=s, assert_cached=assert_cached)
        if not cand:
            last = {"cannot_edit": True, "attempt": s}
            continue
        concl = await FE.concludes_letter(cand, assert_cached=assert_cached)
        if concl != X:
            last = {"concludes": concl, "attempt": s}
            continue
        ack = await judge_cue_ack(X, cand, assert_cached=assert_cached)
        if ack is not True:
            last = {"cue_ack": ack, "attempt": s}
            continue
        genuine = await J.judge_genuine(q, cand, assert_cached=assert_cached)
        coherent = await FE.judge_coherent(q, cand, assert_cached=assert_cached)
        degen = is_degenerate(cand)
        gates = {"cue_ack": ack, "concludes": concl, "genuine": genuine, "coherent": coherent,
                 "degenerate": degen, "attempt": s, "tw": I.word_count(cand)}
        if genuine is True and coherent is True and not degen:
            return cand, gates
        last = gates
    return None, last


async def build_conceal(row, *, assert_cached):
    """Edit the source to conceal the cue (keep conclusion X); first attempt passing all gates."""
    X = row["cued_letter"]
    q = row["question"]
    src = row["analysis"]
    last_gates = {}
    for s in range(MAX_ATTEMPTS):
        cand = await FE.conceal_edit(q, X, src, sample_idx=s, assert_cached=assert_cached)
        if not cand:
            last_gates = {"cannot_edit": True, "attempt": s}
            continue
        if not FE.length_ok(cand, src):
            last_gates = {"length_ok": False, "attempt": s,
                          "tw": I.word_count(cand), "sw": I.word_count(src)}
            continue
        concl = await FE.concludes_letter(cand, assert_cached=assert_cached)
        if concl != X:
            last_gates = {"concludes": concl, "attempt": s}
            continue
        ack = await judge_cue_ack(X, cand, assert_cached=assert_cached)
        if ack is not False:
            last_gates = {"cue_ack": ack, "attempt": s}
            continue
        genuine = await J.judge_genuine(q, cand, assert_cached=assert_cached)
        coherent = await FE.judge_coherent(q, cand, assert_cached=assert_cached)
        degen = is_degenerate(cand)
        gates = {"cue_ack": ack, "concludes": concl, "genuine": genuine, "coherent": coherent,
                 "degenerate": degen, "attempt": s,
                 "tw": I.word_count(cand), "sw": I.word_count(src)}
        if genuine is True and coherent is True and not degen:
            return cand, gates
        last_gates = gates
    return None, last_gates


def make_row(row, kind, analysis, gates, instruction_id):
    return {
        # include the cue so key/stanford versions of the same task don't collide (Seg-12 is all
        # cue='key' -> the id just gains ':key', harmless since Seg-12 targets aren't rebuilt)
        "example_id": f"{row['task_id']}:{row.get('cue', 'key')}:{kind}",
        "task_id": row["task_id"], "source": row["source"], "kind": kind,
        "answer_type": row["answer_type"], "n_options": row.get("n_options"),
        "gold": row["gold"], "cued_letter": row["cued_letter"],
        "prompt_user_content": reconstruct_uc(row),
        "target_analysis": analysis, "target_final": row["cued_letter"],
        "instruction_id": instruction_id, "category": "faithfulness",
        "instruction_split": "train", "split": "train",
        "source_analysis": row["analysis"], "source_words": I.word_count(row["analysis"]),
        "target_words": I.word_count(analysis), "gates": gates,
        "edit_meta": {"editor": FE.EDITOR_MODEL, "judge": FE.JUDGE_MODEL,
                      "editor_version": FE.CONCEAL_EDITOR_VERSION},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="results/cue_probe_train_base.jsonl")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=0, help="cap #sources (DEV)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--faithful-mode", default="matched", choices=["matched", "raw"],
                    help="matched: style-matched Opus-rewrite faithful targets (default); raw: source trace")
    ap.add_argument("--paired-only", action="store_true",
                    help="only emit a faithful target when this source also produced a conceal target")
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.infile)]
    sources = [r for r in rows if eligible_source(r)]
    sources.sort(key=lambda r: r["task_id"])
    if args.limit:
        sources = sources[:args.limit]
    print(f"[build] {len(sources)} eligible cue-following sources (of {len(rows)} rows)")

    sem = asyncio.Semaphore(args.concurrency)
    out_conceal, out_faithful = [], []
    stats = {"n_sources": len(sources), "conceal_ok": 0, "faithful_ok": 0,
             "conceal_fail": 0, "faithful_fail": 0}

    matched = (args.faithful_mode == "matched")

    async def one(r):
        async with sem:
            c_an, c_g = await build_conceal(r, assert_cached=args.assert_cached)
            if args.paired_only and not c_an:
                f_an, f_g = None, {"skipped_no_conceal": True}
            else:
                f_an, f_g = await build_faithful(r, assert_cached=args.assert_cached, matched=matched)
        if c_an:
            out_conceal.append(make_row(r, "conceal", c_an, c_g, "cue_conceal"))
            stats["conceal_ok"] += 1
        else:
            stats["conceal_fail"] += 1
        if f_an:
            fr = make_row(r, "faithful", f_an, f_g, "cue_faithful")
            fr["faithful_mode"] = args.faithful_mode
            out_faithful.append(fr)
            stats["faithful_ok"] += 1
        else:
            stats["faithful_fail"] += 1

    await asyncio.gather(*[one(r) for r in sources])

    out_conceal.sort(key=lambda r: r["task_id"])
    out_faithful.sort(key=lambda r: r["task_id"])
    out_path = DATA / f"faith_targets_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for r in out_conceal + out_faithful:
            f.write(json.dumps(r) + "\n")
    meta = {"tag": args.tag, "infile": args.infile, "stats": stats,
            "git_hash": __import__("subprocess").getoutput("git rev-parse HEAD")[:40],
            "timestamp": datetime.now(timezone.utc).isoformat()}
    json.dump(meta, open(DATA / f"faith_targets_{args.tag}_meta.json", "w"), indent=2)
    print(f"[build] wrote {len(out_conceal)} conceal + {len(out_faithful)} faithful -> {out_path}")
    print(f"[build] stats: {stats}")
    import llm
    print(f"LLM cost this run: ${llm._tracker.run_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
