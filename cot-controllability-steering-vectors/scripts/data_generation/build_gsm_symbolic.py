"""Build a fresh / contamination-robust task set from GSM-Symbolic (Segment 9, Priority 4).

GSM-Symbolic (apple/GSM-Symbolic) = templated GSM8K variants with fresh entities/numbers (+ a canary
string), so "accuracy preserved" on it is stronger evidence the *reasoning* still works (not memorized
public-benchmark answers). Deterministic source-stratified sample; gold = the integer after '####'.
Schema matches data/tasks_all.jsonl (so the steered-eval machinery reuses unchanged).

Usage: python build_gsm_symbolic.py --n 150
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

OUT = Path("data/gsm_symbolic_tasks.jsonl")


def gold_from_answer(ans: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+)", ans)
    return m.group(1).replace(",", "") if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--config", default="main")  # main | p1 | p2 (harder)
    args = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", name=args.config, split="test")
    rows = []
    for ex in ds:
        gold = gold_from_answer(ex["answer"])
        if not gold:
            continue
        rows.append({
            "task_id": f"gsmsym_{args.config}_{ex['id']}_{ex['instance']}",
            "source": "gsm_symbolic", "question": ex["question"].strip(),
            "answer": gold, "answer_type": "number", "n_options": None,
            "split": "fresh", "category": "math_word", "level": None, "subsource": args.config,
            "original_id": ex.get("original_id"),
        })
    random.Random(f"gsm_symbolic:{args.config}").shuffle(rows)
    rows = rows[: args.n]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} GSM-Symbolic ({args.config}) tasks to {OUT}")
    print("example:", json.dumps(rows[0], indent=1)[:500])


if __name__ == "__main__":
    main()
