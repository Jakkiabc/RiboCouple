#!/usr/bin/env python3
"""evaluate_refold.py — refold decoded sequences and compare to target backbones.

Closes the sample/decode loop: folds each designed sequence with RhoFold
(vendored under ribocouple.tools.rhofold_api) and scores the prediction
against the original backbone PDB with the bundled TMscore binary.

Usage:
  python evaluate_refold.py \
      --backbone_dir generated_backbones/ \
      --fasta_dir decoded_sequences/ \
      --rhofold_ckpt /path/to/RhoFold_pretrained.pt \
      --out_dir refold_eval/

Matching is by file stem: backbone_dir/<name>.pdb pairs with the <name>.fasta
sitting next to it (as written by sample_backbones.py) or with
<name>.fasta in --fasta_dir. Directories are searched recursively. With
--pred_dir instead, folding is skipped and pre-computed folded PDBs
(<name>.pdb, any external predictor) are scored directly.

Outputs in --out_dir:
  fastas/, preds/               staged inputs and RhoFold predictions
  refold_per_design.csv         per-design TM-score / RMSD / GDT and validity flags
  refold_summary.json           means, medians, and valid ratios at the
                                --tm_threshold / --rmsd_threshold cutoffs
                                (paper defaults: TM >= 0.45 and RMSD <= 5.5)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("RIBOCOUPLE_SRC", os.path.join(_HERE, "..", "src")))

from ribocouple.analysis.validation_eval import _run_tmscore, _seqrecord_to_str


def _find_by_stem(directory, stem, exts):
    for ext in exts:
        cand = os.path.join(directory, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def _default_tmscore_bin():
    name = "TMscore_macos" if sys.platform == "darwin" else "TMscore"
    return os.path.join(_HERE, "..", "tools", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone_dir", required=True,
                    help="directory of target backbone PDB files")
    ap.add_argument("--fasta_dir", default=None,
                    help="directory of decoded-sequence FASTAs (default: next to each PDB)")
    ap.add_argument("--pred_dir", default=None,
                    help="directory of pre-folded PDBs; skips RhoFold entirely")
    ap.add_argument("--rhofold_ckpt", default=None,
                    help="path to RhoFold_pretrained.pt (required unless --pred_dir)")
    ap.add_argument("--tmscore_bin", default=None,
                    help="TMscore binary (default: bundled tools/TMscore[_macos])")
    ap.add_argument("--device", default="cuda", help="RhoFold device ('cuda' or 'cpu')")
    ap.add_argument("--tm_threshold", type=float, default=0.45)
    ap.add_argument("--rmsd_threshold", type=float, default=5.5)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.pred_dir is None and args.rhofold_ckpt is None:
        ap.error("--rhofold_ckpt is required unless --pred_dir is given")

    tmscore_bin = args.tmscore_bin or _default_tmscore_bin()
    if not os.path.isfile(tmscore_bin):
        ap.error(f"TMscore binary not found: {tmscore_bin}")

    os.makedirs(args.out_dir, exist_ok=True)
    fasta_dir_out = os.path.join(args.out_dir, "fastas")
    pred_dir_out = os.path.join(args.out_dir, "preds")
    os.makedirs(fasta_dir_out, exist_ok=True)
    os.makedirs(pred_dir_out, exist_ok=True)

    backbones = sorted(
        str(p) for p in Path(args.backbone_dir).rglob("*.pdb")
    )
    if not backbones:
        ap.error(f"no .pdb files under {args.backbone_dir}")

    pred_index = None
    if args.pred_dir is not None:
        pred_index = {p.stem: str(p) for p in Path(args.pred_dir).rglob("*.pdb")}

    model = None
    get_features = None
    device = None
    if args.pred_dir is None:
        import torch
        from ribocouple.tools.rhofold_api.rhofold.config import rhofold_config
        from ribocouple.tools.rhofold_api.rhofold.rf import RhoFold
        from ribocouple.tools.rhofold_api.rhofold.utils.alphabet import get_features as _gf

        device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        print(f"Loading RhoFold from {args.rhofold_ckpt} on {device} ...")
        model = RhoFold(rhofold_config)
        state = torch.load(args.rhofold_ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state["model"])
        model.eval().to(device)
        get_features = _gf

    rows, errors = [], []
    for pdb_path in backbones:
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        try:
            if args.pred_dir is not None:
                pred_pdb = pred_index.get(stem)
                if pred_pdb is None:
                    raise FileNotFoundError(f"no folded PDB for {stem} in {args.pred_dir}")
            else:
                sibling = pdb_path[:-4] + ".fasta"
                fasta_src = (
                    sibling if os.path.isfile(sibling) else None
                ) or (
                    _find_by_stem(args.fasta_dir, stem, [".fasta", ".fa"])
                    if args.fasta_dir else None
                )
                if fasta_src is None:
                    raise FileNotFoundError(f"no FASTA for {stem}")

                from Bio import SeqIO
                seq = _seqrecord_to_str(SeqIO.read(fasta_src, "fasta"))
                fasta_path = os.path.join(fasta_dir_out, f"{stem}.fasta")
                with open(fasta_path, "w") as fh:
                    fh.write(f">{stem}\n{seq}\n")

                pred_pdb = os.path.join(pred_dir_out, f"{stem}_rhofold.pdb")
                if not os.path.isfile(pred_pdb):
                    import torch
                    with torch.no_grad():
                        feats = get_features(fasta_path, fasta_path)
                        outputs = model(
                            tokens=feats["tokens"].to(device),
                            rna_fm_tokens=feats["rna_fm_tokens"].to(device),
                            seq=feats["seq"],
                        )
                    out = outputs[-1]
                    coords = out["cord_tns_pred"][-1].squeeze(0).detach().cpu().numpy()
                    conf = out["plddt"][0].detach().cpu().numpy()
                    model.structure_module.converter.export_pdb_file(
                        feats["seq"], coords, path=pred_pdb,
                        chain_id=None, confidence=conf, logger=None,
                    )

            scores = _run_tmscore(tmscore_bin, pdb_path, pred_pdb)
            tm, rmsd = scores["tm_score"], scores["rmsd"]
            rows.append({
                "backbone": pdb_path,
                "pred_pdb": pred_pdb,
                "tm_score": tm,
                "rmsd": rmsd,
                "gdt_ts": scores["gdt_ts"],
                "gdt_ha": scores["gdt_ha"],
                "tm_valid": int(tm >= args.tm_threshold),
                "rmsd_valid": int(rmsd is not None and rmsd <= args.rmsd_threshold),
                "joint_valid": int(tm >= args.tm_threshold
                                   and rmsd is not None and rmsd <= args.rmsd_threshold),
            })
            print(f"{stem}: TM={tm:.3f} RMSD={rmsd:.2f}")
        except Exception as exc:
            errors.append({"backbone": pdb_path, "error": str(exc)})
            print(f"{stem}: FAILED ({exc})", file=sys.stderr)

    per_design_csv = os.path.join(args.out_dir, "refold_per_design.csv")
    if rows:
        with open(per_design_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if errors:
        with open(os.path.join(args.out_dir, "refold_errors.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["backbone", "error"])
            writer.writeheader()
            writer.writerows(errors)

    tms = sorted(r["tm_score"] for r in rows)
    rmsds = sorted(r["rmsd"] for r in rows if r["rmsd"] is not None)
    n = len(rows)
    summary = {
        "n_backbones": len(backbones),
        "n_scored": n,
        "n_errors": len(errors),
        "tm_threshold": args.tm_threshold,
        "rmsd_threshold": args.rmsd_threshold,
        "tm_mean": sum(tms) / n if n else 0.0,
        "tm_median": tms[n // 2] if n else 0.0,
        "rmsd_mean": sum(rmsds) / len(rmsds) if rmsds else None,
        "rmsd_median": rmsds[len(rmsds) // 2] if rmsds else None,
        "tm_valid_ratio": sum(r["tm_valid"] for r in rows) / n if n else 0.0,
        "rmsd_valid_ratio": sum(r["rmsd_valid"] for r in rows) / n if n else 0.0,
        "joint_valid_ratio": sum(r["joint_valid"] for r in rows) / n if n else 0.0,
    }
    with open(os.path.join(args.out_dir, "refold_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Refold summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Per-design CSV: {per_design_csv}")


if __name__ == "__main__":
    main()
