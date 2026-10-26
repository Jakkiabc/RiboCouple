#!/usr/bin/env python3
"""score_pair_confidence.py — score backbone PDBs with the frozen pair head.

Single entry point for pair-confidence scoring. All scoring logic lives in
ribocouple.analysis.pair_rerank_utils.compute_pair_scores; this script only
handles IO so that every consumer (new backbone pools, external generators)
gets bit-identical scores.

Usage:
  python score_pair_confidence.py --ckpt rb_pairtype_e198.ckpt \
      --pdb_dir backbones/ --out pair_scores.csv

Output CSV columns: target_id, length, topk_pair_sum, pair_prob_mean,
pair_prob_max, n_pairs_pred, pair_entropy, pair_prob_std.
Rows are sorted by topk_pair_sum (descending) when the run completes;
--select-top N additionally writes the top-N shortlist to <out>.topN.csv.
Re-running with the same --out skips targets already present (resume).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("RIBOCOUPLE_SRC", os.path.join(_HERE, "..", "src")))

import numpy as np

from ribocouple.analysis.pair_rerank_utils import (
    load_flow_module,
    run_model_on_pdb,
    compute_pair_scores,
)

SCORE_KEYS = [
    "topk_pair_sum",
    "pair_prob_mean",
    "pair_prob_max",
    "n_pairs_pred",
    "pair_entropy",
    "pair_prob_std",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="pair-type FlowModule checkpoint (.ckpt)")
    ap.add_argument("--pdb_dir", required=True, help="directory of backbone PDB files")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--device", default="cuda", help="'cuda' or 'cpu' (falls back to cpu)")
    ap.add_argument("--topk", type=int, default=10, help="top-K pairs for topk_pair_sum")
    ap.add_argument("--t", type=float, default=1.0, help="diffusion timestep for the forward pass")
    ap.add_argument("--select-top", type=int, default=0,
                    help="also write a ranked shortlist of the top N targets to <out>.topN.csv")
    args = ap.parse_args()

    import torch
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    model = load_flow_module(args.ckpt)

    pdb_dir = Path(args.pdb_dir)
    pdbs = sorted(
        p for p in pdb_dir.rglob("*.pdb") if not p.name.startswith("._")
    )
    print(f"[targets] {len(pdbs)} PDBs in {args.pdb_dir}", flush=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["target_id"] for r in csv.DictReader(f)}

    write_header = not os.path.exists(args.out)
    failed = []
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["target_id", "length"] + SCORE_KEYS)
        for i, p in enumerate(pdbs, 1):
            stem = p.relative_to(pdb_dir).with_suffix("").as_posix()
            if stem in done:
                continue
            path = str(p)
            try:
                pair_logits = run_model_on_pdb(model, path, t=args.t, device=device)
                # pdb_to_trans_rot (used by run_model_on_pdb) always returns an
                # all-ones mask, so the mask here is exact.
                res_mask = np.ones(pair_logits.shape[0], dtype=bool)
                scores = compute_pair_scores(pair_logits, res_mask, topk=args.topk)
                w.writerow([stem, pair_logits.shape[0]] +
                           [repr(scores[k]) for k in SCORE_KEYS])
                f.flush()
            except Exception as e:
                failed.append((stem, str(e)))
                print(f"[fail] {stem}: {e}", file=sys.stderr, flush=True)
            if i % 50 == 0:
                print(f"[progress] {i}/{len(pdbs)}", flush=True)

    print(f"[done] {len(pdbs) - len(done) - len(failed)} scored, "
          f"{len(done)} resumed, {len(failed)} failed -> {args.out}", flush=True)

    # Rank by the paper's prioritization score and rewrite the CSV in that order.
    header = ["target_id", "length"] + SCORE_KEYS
    with open(args.out) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["topk_pair_sum"]), reverse=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    if args.select_top > 0:
        top_path = (args.out[:-4] if args.out.endswith(".csv") else args.out) \
                   + f".top{args.select_top}.csv"
        with open(top_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows[:args.select_top])
        print(f"[top] wrote {min(args.select_top, len(rows))} targets -> {top_path}", flush=True)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
