#!/usr/bin/env python3
"""evaluate_inverse_folding.py — recovery and 2D agreement for fixed-backbone designs.

Scores designed sequences against their native backbones: native-sequence
recovery plus secondary-structure agreement (2D MCC and BP precision /
recall / F1). Predicted pairs come from the ViennaRNA MFE fold of each
designed sequence. No refolding is involved.

Two ground-truth tracks, matching the manuscript:
  - canonical track: x3dna find_pair (paper default) or DSSR on each native
    PDB, filtered to canonical Watson-Crick + wobble pairs -> 2D MCC, BP-F1.
  - hbond track: pure-Python structural H-bond pairs (includes non-canonical
    pairs) -> BP-F1_hbond. Always computed; needs no external software.

Requires: the ViennaRNA python bindings (conda: viennarna) and, for the
canonical track only, an annotation binary ($X3DNA/bin/find_pair,
$X3DNA_BIN, or --x3dna_bin; $DSSR_BIN or x3dna-dssr for --pairs_engine
dssr). With --pairs_engine none the canonical track is skipped entirely.

Usage:
  python evaluate_inverse_folding.py \
      --pdb_dir native_pdbs/ \
      --fasta_dir designed_fastas/ \
      --out_dir if_eval/

Matching is by file stem (<name>.pdb with <name>.fasta); directories are
searched recursively. Single-pair mode: --pdb target.pdb --fasta seq.fasta.

Outputs in --out_dir:
  inverse_folding_per_target.csv   per-target recovery / MCC / BP-F1 / counts
  inverse_folding_summary.json     means and medians over all targets
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("RIBOCOUPLE_SRC", os.path.join(_HERE, "..", "src")))

from ribocouple.analysis.inverse_folding_metrics import evaluate_design

METRIC_KEYS = ["recovery", "mcc", "bp_precision", "bp_recall", "bp_f1",
               "bp_hbond_precision", "bp_hbond_recall", "bp_hbond_f1",
               "native_mcc"]


def _read_fasta_seq(path):
    seq_lines = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    if not seq_lines:
        raise ValueError(f"empty FASTA: {path}")
    return "".join(seq_lines)


def _mean_median(values):
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float("nan"), float("nan")
    ordered = sorted(values)
    return sum(values) / len(values), ordered[len(values) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", default=None, help="single native PDB")
    ap.add_argument("--fasta", default=None, help="single designed-sequence FASTA")
    ap.add_argument("--pdb_dir", default=None, help="directory of native PDBs")
    ap.add_argument("--fasta_dir", default=None, help="directory of designed FASTAs")
    ap.add_argument("--x3dna_bin", default=None,
                    help="path to the pair-annotation binary (default: $X3DNA_BIN or "
                         "$X3DNA/bin/find_pair for find_pair; $DSSR_BIN or x3dna-dssr for dssr)")
    ap.add_argument("--pairs_engine", choices=["find_pair", "dssr", "none"],
                    default="find_pair",
                    help="ground-truth pair annotation backend; 'find_pair' (x3dna) matches "
                         "the manuscript, 'dssr' uses x3dna-dssr --json, 'none' skips the "
                         "canonical track (recovery + BP-F1_hbond only; no annotation "
                         "binary required)")
    ap.add_argument("--recovery-only", action="store_true",
                    help="only compute native-sequence recovery; no annotation software "
                         "or ViennaRNA required (skips 2D MCC / BP-F1)")
    ap.add_argument("--no-native-mcc", action="store_true",
                    help="skip the native-sequence ViennaRNA MCC sanity column")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.pdb and args.fasta:
        pairs = [(Path(args.pdb).stem, str(args.pdb), str(args.fasta))]
    elif args.pdb_dir and args.fasta_dir:
        fasta_index = {}
        for ext in ("*.fasta", "*.fa"):
            for p in Path(args.fasta_dir).rglob(ext):
                fasta_index[p.stem] = str(p)
        pairs = []
        for pdb in sorted(Path(args.pdb_dir).rglob("*.pdb")):
            fasta = fasta_index.get(pdb.stem)
            if fasta is None:
                print(f"[warn] no designed FASTA for {pdb.stem}, skipping", file=sys.stderr)
                continue
            pairs.append((pdb.stem, str(pdb), fasta))
        if not pairs:
            ap.error(f"no matched PDB/FASTA pairs under {args.pdb_dir} and {args.fasta_dir}")
    else:
        ap.error("give either --pdb/--fasta or --pdb_dir/--fasta_dir")

    os.makedirs(args.out_dir, exist_ok=True)
    rows, errors = [], []
    for target_id, pdb_path, fasta_path in pairs:
        try:
            seq = _read_fasta_seq(fasta_path)
            row = evaluate_design(
                pdb_path, seq,
                find_pair_bin=args.x3dna_bin,
                native_mcc=not args.no_native_mcc,
                pairs_engine=args.pairs_engine,
                recovery_only=args.recovery_only,
            )
            row_out = {"target_id": target_id}
            row_out.update(row)
            rows.append(row_out)
            if args.recovery_only:
                print(f"{target_id}: recovery={row['recovery']:.3f}")
            elif args.pairs_engine == "none":
                print(f"{target_id}: recovery={row['recovery']:.3f} "
                      f"BP-F1_hbond={row['bp_hbond_f1']:.3f}")
            else:
                print(f"{target_id}: recovery={row['recovery']:.3f} "
                      f"MCC={row['mcc']:.3f} BP-F1={row['bp_f1']:.3f} "
                      f"BP-F1_hbond={row['bp_hbond_f1']:.3f}")
        except Exception as exc:
            hint = (" [hint: no pair-annotation binary found; use --recovery-only "
                    "to skip 2D metrics]"
                    if isinstance(exc, FileNotFoundError) else "")
            errors.append({"target_id": target_id, "error": str(exc)})
            print(f"{target_id}: FAILED ({exc}){hint}", file=sys.stderr)

    per_target_csv = os.path.join(args.out_dir, "inverse_folding_per_target.csv")
    if rows:
        fieldnames = ["target_id", "length", "n_true_pairs", "n_pred_pairs",
                      "n_hbond_pairs", "tp", "fp", "fn", "tn", "recovery", "mcc",
                      "bp_precision", "bp_recall", "bp_f1",
                      "bp_hbond_precision", "bp_hbond_recall", "bp_hbond_f1",
                      "native_mcc"]
        fieldnames = [f for f in fieldnames if f in rows[0]]
        with open(per_target_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if errors:
        with open(os.path.join(args.out_dir, "inverse_folding_errors.csv"),
                  "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["target_id", "error"])
            writer.writeheader()
            writer.writerows(errors)

    summary = {"n_targets": len(rows), "n_errors": len(errors),
               "pairs_engine": args.pairs_engine}
    for key in METRIC_KEYS:
        if rows and key in rows[0]:
            mean, median = _mean_median([r[key] for r in rows])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_median"] = median
    with open(os.path.join(args.out_dir, "inverse_folding_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Inverse-folding summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Per-target CSV: {per_target_csv}")


if __name__ == "__main__":
    main()
