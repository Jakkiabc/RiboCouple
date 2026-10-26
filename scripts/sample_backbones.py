#!/usr/bin/env python3
"""Unconditionally sample RNA backbones from a trained RiboCouple checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Trained RiboCouple Lightning checkpoint")
    parser.add_argument("--out-dir", required=True, help="Directory for generated PDB files and manifest")
    parser.add_argument("--lengths", nargs="+", type=int, default=list(range(50, 151, 10)))
    parser.add_argument("--samples-per-length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None,
                        help="Sampling steps; defaults to the checkpoint config value")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-sequences", action="store_true",
                        help="do not write per-sample FASTAs even if the checkpoint has a sequence head")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_length < 1:
        raise ValueError("--samples-per-length must be positive")
    if any(length < 1 for length in args.lengths):
        raise ValueError("All requested lengths must be positive")

    import numpy as np
    import torch
    from ribocouple.analysis import utils as analysis_utils
    from ribocouple.models.checkpoint import load_flow_module
    from ribocouple.data import utils as data_utils

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.csv").exists() or (out_dir / "run.json").exists():
        raise FileExistsError(
            f"Sampling output already exists in {out_dir}; choose a new --out-dir "
            "so the manifest cannot refer to overwritten PDB files."
        )

    module = load_flow_module(args.ckpt, cfg=None).to(device).eval()
    interpolant = module.interpolant
    if args.steps is not None:
        if args.steps < 2:
            raise ValueError("--steps must be at least 2")
        interpolant._sample_cfg.num_timesteps = args.steps
    steps = int(interpolant._sample_cfg.num_timesteps)
    interpolant.set_device(device)

    manifest_path = out_dir / "manifest.csv"
    fields = ["length", "sample_index", "seed", "steps", "pdb", "fasta"]
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for length in args.lengths:
            length_dir = out_dir / f"length_{length}"
            length_dir.mkdir(exist_ok=True)
            for sample_index in range(args.samples_per_length):
                # Matching length/index pairs use the same initial RNG seed across step settings.
                sample_seed = args.seed + length * 100_000 + sample_index
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32))
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

                with torch.inference_mode():
                    if args.no_sequences:
                        atom37_traj, _, _ = interpolant.sample(1, length, module.model)
                        seq_logits = None
                    else:
                        atom37_traj, _, _, model_out = interpolant.sample(
                            1, length, module.model, return_model_out=True
                        )
                        seq_logits = model_out.get("seq_logits")
                trajectory = data_utils.to_numpy(torch.concat(atom37_traj, dim=0))
                final_atoms = trajectory[-1]
                target = length_dir / f"sample_{sample_index:03d}"
                na_mask = torch.ones((1, length), dtype=torch.long)
                pdb_path = analysis_utils.write_complex_to_pdbs(
                    final_atoms,
                    str(target),
                    is_na_residue_mask=na_mask,
                )
                fasta_rel = ""
                if seq_logits is not None:
                    seq_indices = torch.argmax(seq_logits[0], dim=-1).cpu().numpy()
                    seq_str = "".join("ACGU"[int(k)] for k in seq_indices if 0 <= int(k) < 4)
                    fasta_path = Path(pdb_path).with_suffix(".fasta")
                    fasta_path.write_text(f">{target.name}\n{seq_str}\n")
                    fasta_rel = str(fasta_path.relative_to(out_dir))
                writer.writerow({
                    "length": length,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "steps": steps,
                    "pdb": str(Path(pdb_path).relative_to(out_dir)),
                    "fasta": fasta_rel,
                })
                handle.flush()
                print(f"[{length} nt | {sample_index + 1}/{args.samples_per_length}] {pdb_path}", flush=True)

    (out_dir / "run.json").write_text(json.dumps({
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "lengths": args.lengths,
        "samples_per_length": args.samples_per_length,
        "sampling_steps": steps,
        "base_seed": args.seed,
        "device": str(device),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
