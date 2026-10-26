#!/usr/bin/env python3
"""run_generation_pipeline.py — unconditional generation, end to end.

One command for the generation workflow: sample backbones (with sequences
when the checkpoint has a sequence head), score every sample with the pair
head, select the top-K per length by topk_pair_sum, and refold only the
selected candidates with RhoFold + TMscore. An optional random-K baseline
per length quantifies the reranking gain. Pair scoring is cheap and covers
the whole pool; RhoFold runs only on the selected subset.

Usage:
  python run_generation_pipeline.py \
      --ckpt model.ckpt \
      --out_dir pipeline_out/ \
      --lengths 50 60 70 80 90 100 110 120 130 140 150 \
      --samples-per-length 50 \
      --top-k 5 --random-k \
      --rhofold_ckpt /path/to/RhoFold_pretrained.pt

With --refold-all instead of --top-k, reranking is skipped and every sampled
backbone is refolded (full-pool evaluation, no selection).

Outputs in --out_dir:
  generated_pdbs/len{L}_sample{i}.pdb/.fasta   sampled backbones + sequences
  pair_scores.csv                              six pair scores per sample
  rerank_selections.csv                        top-K / random-K picks
  external_eval/<tag>/                         RhoFold preds + TMscore tables
  pipeline_summary.csv/json                    refold metrics per selection
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("RIBOCOUPLE_SRC", os.path.join(_HERE, "..", "src")))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="RiboCouple Lightning checkpoint")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=list(range(50, 151, 10)))
    ap.add_argument("--samples-per-length", type=int, default=50)
    ap.add_argument("--steps", type=int, default=None,
                    help="sampling steps; defaults to the checkpoint config value")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=5,
                    help="candidates per length selected by topk_pair_sum")
    ap.add_argument("--random-k", action="store_true",
                    help="also evaluate a random-K baseline per length")
    ap.add_argument("--refold-all", action="store_true",
                    help="skip reranking and refold every sampled backbone")
    ap.add_argument("--rhofold_ckpt", default=None,
                    help="RhoFold_pretrained.pt; omit to skip refolding")
    ap.add_argument("--tmscore_bin", default=None,
                    help="TMscore binary (default: bundled tools/TMscore[_macos])")
    ap.add_argument("--tm_threshold", type=float, default=0.45)
    ap.add_argument("--rmsd_threshold", type=float, default=5.5)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import numpy as np
    import pandas as pd
    import torch

    from ribocouple.analysis.pair_rerank_utils import (
        load_flow_module,
        generate_samples,
        run_model_on_atom37,
        compute_pair_scores,
        write_atom37_to_pdb,
    )

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU", file=sys.stderr)
        device = "cpu"

    # 1. load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = load_flow_module(args.ckpt, cfg=None).to(device).eval()
    cfg = model.hparams.get("cfg", None)
    if cfg is None:
        raise ValueError("Config not found in checkpoint.")
    if args.steps is not None:
        cfg.interpolant.sampling.num_timesteps = args.steps

    # 2. sample backbones + sequences
    print(f"Sampling {args.samples_per_length} backbones per length: {args.lengths}")
    samples = generate_samples(
        model=model, cfg=cfg, lengths=args.lengths,
        n_samples_per_length=args.samples_per_length,
        device=device, seed=args.seed, return_sequences=True,
    )
    if "seq_indices" not in samples[0]:
        if args.rhofold_ckpt:
            raise ValueError(
                "This checkpoint produced no sequences (no sequence head), "
                "so there is nothing to refold. Use a pair-type/sequence-head "
                "checkpoint, or drop --rhofold_ckpt and decode sequences "
                "separately with decode_pair_coupled.py."
            )
        print("[warn] checkpoint has no sequence head; writing PDBs only",
              file=sys.stderr)

    # 3. write PDB/FASTA, score with the pair head
    pdb_dir = os.path.join(args.out_dir, "generated_pdbs")
    os.makedirs(pdb_dir, exist_ok=True)
    records = []
    for sample in samples:
        length, idx = sample["length"], sample["sample_idx"]
        name = f"len{length}_sample{idx}"
        pdb_path = write_atom37_to_pdb(sample["atom37"],
                                       os.path.join(pdb_dir, f"{name}.pdb"))
        if "seq_indices" in sample:
            seq = "".join("ACGU"[int(k)] for k in sample["seq_indices"]
                          if 0 <= int(k) < 4)
            with open(pdb_path[:-4] + ".fasta", "w") as fh:
                fh.write(f">{name}\n{seq}\n")
        outputs = run_model_on_atom37(
            model, sample["atom37"], sample["res_mask"], t=1.0, device=device
        )
        row = {"length": length, "sample_idx": idx, "pdb_path": pdb_path}
        row.update(compute_pair_scores(outputs["pair_logits"], sample["res_mask"], topk=10))
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("topk_pair_sum", ascending=False)
    scores_csv = os.path.join(args.out_dir, "pair_scores.csv")
    df.to_csv(scores_csv, index=False)
    print(f"Pair scores for {len(df)} samples -> {scores_csv}")

    # 4. select top-K (and random-K) per length, or keep everything
    selections = []
    for length in args.lengths:
        sub = df[df["length"] == length]
        if len(sub) == 0:
            continue
        if args.refold_all:
            for _, r in sub.iterrows():
                selections.append({"length": length, "rank_type": "all",
                                   "pdb_path": r["pdb_path"],
                                   "topk_pair_sum": r["topk_pair_sum"]})
            continue
        k = min(args.top_k, len(sub))
        top = sub.head(k)
        for _, r in top.iterrows():
            selections.append({"length": length, "rank_type": "top",
                               "pdb_path": r["pdb_path"],
                               "topk_pair_sum": r["topk_pair_sum"]})
        if args.random_k:
            rand = sub.sample(n=k, random_state=args.seed)
            for _, r in rand.iterrows():
                selections.append({"length": length, "rank_type": "random",
                                   "pdb_path": r["pdb_path"],
                                   "topk_pair_sum": r["topk_pair_sum"]})
    sel_df = pd.DataFrame(selections)
    sel_csv = os.path.join(args.out_dir, "rerank_selections.csv")
    sel_df.to_csv(sel_csv, index=False)
    print(f"Selections -> {sel_csv}")

    # 5. refold selected candidates only
    summary_rows = []
    if args.rhofold_ckpt:
        from ribocouple.analysis.validation_eval import ValidationExternalEvaluator
        ext_cfg = SimpleNamespace(
            sequence_source="e4",
            n_seqs_per_backbone=1,
            temperature=0.1,
            tm_threshold=args.tm_threshold,
            rmsd_threshold=args.rmsd_threshold,
            best_by="tm_score",
            rhofold_ckpt=args.rhofold_ckpt,
            tmscore_bin=args.tmscore_bin,
            rhofold_device=device,
            grnade_gpu_id=0,
            repo_root=os.path.abspath(os.path.join(_HERE, "..")),
            enabled=True,
        )
        evaluator = ValidationExternalEvaluator(ext_cfg)
        print("Loading RhoFold (once) ...")
        evaluator._load_models()

        rank_types = ["all"] if args.refold_all else (
            ["top", "random"] if args.random_k else ["top"]
        )
        for rank_type in rank_types:
            for length in args.lengths:
                sel = sel_df[(sel_df["length"] == length)
                             & (sel_df["rank_type"] == rank_type)]
                if len(sel) == 0:
                    continue
                tag = f"len{length}_{rank_type}"
                eval_dir = os.path.join(args.out_dir, "external_eval", tag)
                summary = evaluator.evaluate(sel["pdb_path"].tolist(), eval_dir)
                row = {"length": length, "rank_type": rank_type,
                       "n_samples": len(sel)}
                row.update({k: v for k, v in summary.items()
                            if k.startswith("ext_")})
                summary_rows.append(row)
                print(f"[{tag}] joint-valid={summary.get('ext_joint_valid_ratio', 0):.3f} "
                      f"tm_mean={summary.get('ext_tm_mean', 0):.3f}")
    else:
        print("No --rhofold_ckpt given; skipping refolding. "
              "Rerun evaluate_refold.py on the selections later to add it.")

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        sum_csv = os.path.join(args.out_dir, "pipeline_summary.csv")
        sum_df.to_csv(sum_csv, index=False)
        with open(os.path.join(args.out_dir, "pipeline_summary.json"), "w") as fh:
            json.dump(summary_rows, fh, indent=2, default=float)

        # Overall aggregation weighted by the per-design rows (not by
        # per-length means), reusing the per-length best CSVs.
        best_frames = []
        for tag_dir in sorted(Path(args.out_dir, "external_eval").iterdir()):
            best_csv = tag_dir / "external_eval_best.csv"
            if tag_dir.is_dir() and best_csv.exists():
                frame = pd.read_csv(best_csv)
                frame["length"] = int(tag_dir.name.split("_")[0].replace("len", ""))
                frame["rank_type"] = tag_dir.name.split("_")[1]
                best_frames.append(frame)
        if best_frames:
            all_designs = pd.concat(best_frames, ignore_index=True)
            all_designs.to_csv(os.path.join(args.out_dir, "pipeline_per_design.csv"),
                               index=False)
            overall_rows = []
            for rank_type, g in all_designs.groupby("rank_type"):
                overall_rows.append({
                    "rank_type": rank_type,
                    "n": len(g),
                    "tm_mean": g["tm_score"].mean(),
                    "rmsd_mean": g["rmsd"].mean(),
                    "tm_valid_ratio": g["tm_valid"].mean(),
                    "rmsd_valid_ratio": g["rmsd_valid"].mean(),
                    "joint_valid_ratio": g["joint_valid"].mean(),
                })
            overall = pd.DataFrame(overall_rows)
            overall.to_csv(os.path.join(args.out_dir, "pipeline_overall.csv"),
                           index=False)
            print("\n=== Overall (weighted by designs) ===")
            print(overall.to_string(index=False))
        print(f"\nSummary -> {sum_csv}")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
