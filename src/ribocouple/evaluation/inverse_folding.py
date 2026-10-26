#!/usr/bin/env python3
"""Evaluate the RiboCouple sequence head as an inverse-folding model.

The input is a directory of *real target RNA backbones*.  For every PDB this
script runs the existing checkpoint at t=1, decodes one argmax sequence, folds
that sequence with the project's ValidationExternalEvaluator/RhoFold wrapper,
and compares the folded structure with the same target PDB.

Two inference modes are available without retraining:

  predicted  -- the checkpoint's normal predicted pair bias (main result)
  zero_pair  -- replace the pair bias entering the Transformer sequence head
                by zero (matched topology ablation)

If native bases can be read from the PDB residue names, ``--include_native``
also evaluates the native sequence.  Native is an evaluator calibration, not a
design baseline.

This module also exposes PDB parsing helpers used by the decoding entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from omegaconf import OmegaConf

from ribocouple.models.checkpoint import load_flow_module
from ribocouple.data.pdb_io import RNA_RESNAME, pdb_frames_and_native_sequence
from ribocouple.analysis.utils import write_complex_to_pdbs
from ribocouple.analysis.validation_eval import ValidationExternalEvaluator
from ribocouple.data import utils as data_utils
from ribocouple.data.pdb_na_dataset_base import PDBNABaseDataset


NUCLEOTIDES = "ACGU"  # project order: A=0, C=1, G=2, U=3



def parse_args():
    p = argparse.ArgumentParser(description="Real-backbone RNA inverse-folding evaluation")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_path", default="configs/config.yaml")
    p.add_argument("--input_pdb_dir", default=None,
                   help="Optional directory of real target PDBs. If omitted, "
                        "sample seen training structures from data_cfg.csv_path")
    p.add_argument("--metadata_csv", default=None,
                   help="Override data_cfg.csv_path for automatic training-data mode")
    p.add_argument("--manifest_csv", default=None,
                   help="Optional RCSB external_test_manifest.csv. If omitted, "
                        "auto-detect it beside the input PDB directory")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--decode_modes", nargs="+", choices=["predicted", "zero_pair"],
                   default=["predicted", "zero_pair"])
    p.add_argument("--include_native", action="store_true")
    p.add_argument("--include_grnade", action="store_true",
                   help="Run gRNAde on each unique clean target backbone")
    p.add_argument("--grnade_n_samples", type=int, default=1,
                   help="gRNAde samples per backbone; best TM is reported if >1")
    p.add_argument("--grnade_temperature", type=float, default=0.1)
    p.add_argument("--length_min", type=int, default=1)
    p.add_argument("--length_max", type=int, default=149)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rhofold_device", default="cuda:0")
    p.add_argument(
        "--rhofold_ckpt",
        default=os.path.join(
            os.environ.get("RBD_RHOFOLD_ROOT", "rna-backbone-design/src/tools"),
            "rhofold_api", "Rhofold_pretrained.pt",
        ),
    )
    p.add_argument("--tmscore_bin", default=(
        os.environ.get("TMSCORE_BIN", "TMscore")
    ))
    p.add_argument("--tm_threshold", type=float, default=0.45)
    p.add_argument("--rmsd_threshold", type=float, default=5.5)
    p.add_argument("--skip_rhofold", action="store_true")
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _RNAChainSelect(Select):
    def __init__(self, model_id, chain_id, residue_ids):
        self.model_id = model_id
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)

    def accept_model(self, model):
        return int(model.id == self.model_id)

    def accept_chain(self, chain):
        return int(chain.id == self.chain_id)

    def accept_residue(self, residue):
        return int(residue.id in self.residue_ids)

    def accept_atom(self, atom):
        # Remove alternate conformers except blank/A while retaining all atoms
        # needed by TM-score and frame extraction.
        altloc = atom.get_altloc()
        return int(altloc in (" ", "A", "1"))


def extract_longest_clean_rna_chain(source: Path, destination: Path):
    """Write the longest complete RNA chain from model 0 to a clean PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source.stem, str(source))
    model = next(structure.get_models())
    candidates = []
    for chain in model:
        residues = []
        for residue in chain:
            name = residue.get_resname().strip().upper()
            if name in RNA_RESNAME and all(a in residue for a in ("C3'", "C4'", "O4'")):
                residues.append(residue)
        if residues:
            candidates.append((chain, residues))
    if not candidates:
        raise ValueError("no canonical RNA chain with complete frame atoms")
    chain, residues = max(candidates, key=lambda item: len(item[1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(
        str(destination),
        _RNAChainSelect(model.id, chain.id, [r.id for r in residues]),
    )
    return {
        "selected_chain": chain.id,
        "n_complete_rna_residues": len(residues),
        "n_rna_chains_in_model": len(candidates),
    }


def write_grnade_compatible_pdb(source: Path, destination: Path, sequence: str):
    """Canonicalize chain/residue IDs so gRNAde's cpdb parser keeps all residues."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source.stem, str(source))
    model = next(structure.get_models())
    choices = []
    for chain in model:
        residues = [
            residue for residue in chain
            if all(name in residue for name in ("C3'", "C4'", "O4'"))
        ]
        if residues:
            choices.append((chain, residues))
    if not choices:
        raise ValueError(f"No RNA frame residues in {source}")
    selected_chain, residues = max(choices, key=lambda item: len(item[1]))
    if len(sequence) != len(residues):
        raise ValueError(
            f"gRNAde canonicalization length mismatch for {source.name}: "
            f"sequence={len(sequence)} residues={len(residues)}"
        )

    # Work on a copy, retain one model/chain, and make every residue a standard
    # ATOM record with a unique sequential identifier.  gRNAde's cpdb parser
    # otherwise merges author-numbered or modified residues.
    clean_structure = copy.deepcopy(structure)
    clean_model = next(clean_structure.get_models())
    for chain in list(clean_model):
        if chain.id != selected_chain.id:
            clean_model.detach_child(chain.id)
    chain = clean_model[selected_chain.id]
    kept = [
        residue for residue in chain
        if all(name in residue for name in ("C3'", "C4'", "O4'"))
    ]
    keep_ids = {id(residue) for residue in kept}
    for residue in list(chain):
        if id(residue) not in keep_ids:
            chain.detach_child(residue.id)
    # Temporary unique IDs prevent collisions while replacing author numbering.
    for idx, residue in enumerate(kept):
        residue.id = (" ", -(idx + 1), " ")
    for idx, (residue, base) in enumerate(zip(kept, sequence), start=1):
        residue.resname = base
        residue.id = (" ", idx, " ")
    chain.id = "A"
    destination.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(clean_structure)
    io.save(str(destination))
    return destination


def find_sequence_head(module):
    """Locate the Transformer sequence head used by the loaded checkpoint."""
    candidates = []
    for root in (getattr(module, "model", None), module):
        if root is None:
            continue
        for name in ("seq_pred_net", "sequence_head", "seq_head"):
            head = getattr(root, name, None)
            if head is not None:
                candidates.append(head)
    for head in candidates:
        # The pair-biased head accepts (node_embed, pair_logits, res_mask).
        if hasattr(head, "register_forward_pre_hook"):
            return head
    raise RuntimeError("Could not locate sequence head on loaded checkpoint")


@contextmanager
def pair_input_ablation(module, mode: str):
    """Temporarily replace only the pair tensor entering the sequence head."""
    if mode == "predicted":
        yield
        return
    head = find_sequence_head(module)

    def pre_hook(_head, args):
        if len(args) < 3 or not torch.is_tensor(args[1]):
            raise RuntimeError(
                "zero_pair requires a Transformer sequence head called as "
                "(node_embed, pair_logits, res_mask)"
            )
        return (args[0], torch.zeros_like(args[1]), args[2], *args[3:])

    handle = head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def run_one(module, trans, rotmats, res_mask, mode: str, device: str):
    batch = {
        "res_mask": torch.from_numpy(res_mask).float().unsqueeze(0).to(device),
        "trans_t": torch.from_numpy(trans).float().unsqueeze(0).to(device),
        "rotmats_t": torch.from_numpy(rotmats).float().unsqueeze(0).to(device),
        "t": torch.ones((1, 1), device=device),
    }
    with torch.inference_mode(), pair_input_ablation(module, mode):
        output = module.model(batch)
    seq_logits = output.get("seq_logits")
    if seq_logits is None:
        raise RuntimeError("Checkpoint does not output seq_logits")
    indices = torch.argmax(seq_logits[0], dim=-1).detach().cpu().numpy()
    sequence = "".join(NUCLEOTIDES[int(i)] for i in indices)
    pair_logits = output.get("pair_logits")
    pair_max = float(torch.sigmoid(pair_logits[0]).max().item()) if pair_logits is not None else np.nan
    return sequence, pair_max, seq_logits[0].detach().cpu()


def sequence_metrics(seq_logits: torch.Tensor, native: str):
    """recovery (argmax vs native) + perplexity (CE vs native) from seq_logits."""
    import torch.nn.functional as F
    target = torch.tensor([NUCLEOTIDES.find(b) for b in native], dtype=torch.long)
    valid = target >= 0
    L = min(int(valid.sum().item()) and len(native), seq_logits.shape[0])
    if L == 0 or valid.sum().item() == 0:
        return None, None
    logits = seq_logits[:L]
    tgt = target[:L]
    val = tgt >= 0
    if val.sum().item() == 0:
        return None, None
    pred = logits.argmax(dim=-1)
    rec = (pred[val] == tgt[val]).float().mean().item()
    ce = F.cross_entropy(logits[val], tgt[val], reduction="mean").item()
    return rec, float(np.exp(ce))


def make_evaluator(args, sequence_source="e4", n_sequences=1, temperature=0.1):
    cfg = SimpleNamespace(
        rhofold_device=args.rhofold_device,
        grnade_gpu_id=0,
        n_seqs_per_backbone=n_sequences,
        temperature=temperature,
        tm_threshold=args.tm_threshold,
        rmsd_threshold=args.rmsd_threshold,
        best_by="tm_score",
        rhofold_ckpt=args.rhofold_ckpt,
        tmscore_bin=args.tmscore_bin,
        sequence_source=sequence_source,
        enabled=True,
    )
    evaluator = ValidationExternalEvaluator(cfg)
    print("Loading RhoFold once ...")
    evaluator._load_models()
    return evaluator


def find_external_csv(eval_dir: Path):
    for name in ("external_eval_per_sequence.csv", "external_eval_best.csv"):
        path = eval_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No external evaluation CSV under {eval_dir}")


def standardize_eval(eval_df: pd.DataFrame):
    out = eval_df.copy()
    if "pdb_name" not in out.columns and "generated_pdb" in out.columns:
        out["pdb_name"] = out["generated_pdb"].map(lambda x: Path(str(x)).stem)
    return out


def length_bin(length: int):
    if length < 50:
        return "lt50"
    if length < 80:
        return "50-79"
    if length < 110:
        return "80-109"
    if length < 150:
        return "110-149"
    return "150+"


def load_external_native_manifest(input_pdb_dir: str, explicit_path: Optional[str]):
    """Return PDB-ID -> canonical RNA sequence from the RCSB download manifest."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    root = Path(input_pdb_dir)
    candidates.extend([
        root / "external_test_manifest.csv",
        root.parent / "external_test_manifest.csv",
    ])
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if manifest_path is None:
        return {}, None
    frame = pd.read_csv(manifest_path)
    if not {"pdb_id", "sequence"} <= set(frame.columns):
        raise ValueError(
            f"Manifest {manifest_path} must contain pdb_id and sequence columns"
        )
    mapping = {}
    for row in frame.itertuples(index=False):
        pdb_id = str(getattr(row, "pdb_id")).strip().upper()
        sequence = "".join(str(getattr(row, "sequence")).split()).upper()
        if sequence and set(sequence) <= set(NUCLEOTIDES):
            mapping[pdb_id] = sequence
    return mapping, manifest_path


def _native_from_dataset_item(item: Dict, length: int) -> Optional[str]:
    values = torch.as_tensor(item["aatype"])[:length].cpu().numpy()
    if np.any((values < 0) | (values > 3)):
        return None
    return "".join(NUCLEOTIDES[int(x)] for x in values)


def _real_atom37_from_processed(processed_path: str) -> np.ndarray:
    """Recover modeled RNA atom37 coordinates from a dataset processed pickle."""
    feats = data_utils.read_pkl(processed_path)
    feats = data_utils.parse_complex_feats(feats)
    positions = np.asarray(feats["atom_positions"])
    modeled = np.asarray(feats["modeled_idx"], dtype=np.int64)
    lo, hi = int(modeled.min()), int(modeled.max()) + 1
    positions = positions[lo:hi]

    molecule = np.asarray(feats["molecule_type_encoding"])[lo:hi]
    is_na = (molecule[:, 1] == 1) | (molecule[:, 2] == 1)
    positions = positions[is_na]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"unexpected atom_positions shape {positions.shape}")
    if positions.shape[1] > 37:
        positions = positions[:, :37]
    elif positions.shape[1] < 37:
        padded = np.zeros((positions.shape[0], 37, 3), dtype=positions.dtype)
        padded[:, :positions.shape[1]] = positions
        positions = padded
    return positions.astype(np.float32)


def load_seen_training_targets(cfg, metadata_csv: Optional[str], length_min: int,
                               length_max: int, max_samples: Optional[int], seed: int,
                               target_dir: Path):
    """Sample the exact filtered training pool used by PDBNABaseDataset."""
    if metadata_csv:
        OmegaConf.set_struct(cfg, False)
        cfg.data_cfg.csv_path = metadata_csv
        OmegaConf.set_struct(cfg, True)
    dataset = PDBNABaseDataset(cfg.data_cfg, is_training=True)
    eligible = dataset.csv[
        (dataset.csv["modeled_na_seq_len"] >= length_min)
        & (dataset.csv["modeled_na_seq_len"] <= length_max)
    ]
    if eligible.empty:
        raise RuntimeError("No filtered training structures in requested length range")
    n = len(eligible) if max_samples is None else min(int(max_samples), len(eligible))
    chosen = eligible.sample(n=n, replace=False, random_state=seed)

    targets = []
    # dataset.csv retains original dataframe indices; map them back to iloc safely.
    index_to_iloc = {idx: pos for pos, idx in enumerate(dataset.csv.index.tolist())}
    for order, (csv_index, row) in enumerate(chosen.iterrows()):
        try:
            item = dataset[index_to_iloc[csv_index]]
            length = int(torch.as_tensor(item["res_mask"]).sum().item())
            trans = torch.as_tensor(item["trans_1"])[:length].float().cpu().numpy()
            rotmats = torch.as_tensor(item["rotmats_1"])[:length].float().cpu().numpy()
            mask = np.ones(length, dtype=np.float32)
            native = _native_from_dataset_item(item, length)
            atom37 = _real_atom37_from_processed(str(row["processed_path"]))[:length]
            raw_name = str(row.get("pdb_name", f"train_{order:05d}"))
            target_id = f"seen_{order:05d}_{raw_name}"
            requested = target_dir / f"{target_id}.pdb"
            actual = Path(write_complex_to_pdbs(atom37, str(requested), is_na_residue_mask=np.ones(atom37.shape[0], dtype=bool)))
            targets.append((actual, trans, rotmats, mask, native, target_id))
        except Exception as exc:
            print(f"[skip training row {csv_index}] {row.get('processed_path')}: {exc}")
    if not targets:
        raise RuntimeError("Could not reconstruct any training target structures")
    return targets


def summarize(df: pd.DataFrame, group_cols: List[str]):
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(group)
        for col in ("tm_score", "rmsd", "tm_valid", "rmsd_valid", "joint_valid"):
            if col in group:
                row[f"mean_{col}"] = pd.to_numeric(group[col], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.output_dir)
    target_dir = out / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    cfg = OmegaConf.load(args.config_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.ckpt_path}")
    try:
        # Preferred: the repo config (matches the current FlowModule code).
        module = load_flow_module(args.ckpt_path, cfg=cfg).to(device).eval()
    except RuntimeError as exc:
        # Older variants (v4 binary-pair / nopair) lack pair-type weights; build
        # the FlowModule from the config stored inside the checkpoint instead.
        print(f"[warn] repo-config load failed ({exc}); retrying with ckpt-internal cfg")
        module = load_flow_module(args.ckpt_path, cfg=None).to(device).eval()

    candidates = []
    if args.input_pdb_dir:
        clean_input_dir = out / "clean_input_chains"
        clean_rows = []
        manifest_native, manifest_path = load_external_native_manifest(
            args.input_pdb_dir, args.manifest_csv
        )
        if manifest_path:
            print(f"Using canonical native sequences from: {manifest_path}")
        else:
            print("No external manifest found; native sequences will use PDB residue names")
        for path in sorted(Path(args.input_pdb_dir).rglob("*.pdb")):
            try:
                clean_path = clean_input_dir / f"{path.stem}__clean.pdb"
                clean_info = extract_longest_clean_rna_chain(path, clean_path)
                trans, rotmats, mask, pdb_native = pdb_frames_and_native_sequence(str(clean_path))
            except Exception as exc:
                print(f"[skip] {path}: {exc}")
                continue
            length = int(mask.sum())
            pdb_id = path.stem.split("__", 1)[0].upper()
            manifest_sequence = manifest_native.get(pdb_id)
            native_source = "pdb_residue_names" if pdb_native is not None else "missing"
            native = pdb_native
            if manifest_sequence is not None:
                if len(manifest_sequence) == length:
                    native = manifest_sequence
                    native_source = "rcsb_manifest_canonical_sequence"
                elif native is None:
                    native_source = (
                        f"manifest_length_mismatch:{len(manifest_sequence)}!={length}"
                    )
            if args.length_min <= length <= args.length_max:
                candidates.append((clean_path, trans, rotmats, mask, native, None))
                clean_rows.append({
                    "source_pdb": str(path),
                    "clean_pdb": str(clean_path),
                    "length": length,
                    "native_sequence": native or "",
                    "native_sequence_source": native_source,
                    "manifest_sequence_length": (
                        len(manifest_sequence) if manifest_sequence is not None else ""
                    ),
                    **clean_info,
                })
            else:
                print(f"[skip length={length}] {path}")
        pd.DataFrame(clean_rows).to_csv(out / "clean_input_manifest.csv", index=False)
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        if args.max_samples is not None:
            candidates = candidates[:args.max_samples]
        evaluation_scope = "external_or_heldout_pdb"
    else:
        print(f"Sampling seen training structures from: {args.metadata_csv or cfg.data_cfg.csv_path}")
        candidates = load_seen_training_targets(
            cfg, args.metadata_csv, args.length_min, args.length_max,
            args.max_samples, args.seed, target_dir,
        )
        evaluation_scope = "seen_train_backbone"
    if not candidates:
        raise RuntimeError("No valid target PDBs matched the requested length range")

    rows: List[Dict] = []
    eval_paths: List[str] = []
    unique_target_paths: List[str] = []
    target_metadata: Dict[str, Dict] = {}
    used_ids = set()
    for sample_idx, (source, trans, rotmats, mask, native, preset_id) in enumerate(candidates):
        base = source.stem
        target_id = preset_id or (base if base not in used_ids else f"{base}_{sample_idx:05d}")
        used_ids.add(target_id)
        copied_target = target_dir / f"{target_id}.pdb"
        shutil.copy2(source, copied_target)
        length = int(mask.sum())
        unique_target_paths.append(str(copied_target))
        target_metadata[target_id] = {
            "target_id": target_id,
            "source_pdb": str(source),
            "target_pdb": str(copied_target),
            "length": length,
            "length_bin": length_bin(length),
            "evaluation_scope": evaluation_scope,
            "native_sequence": native,
        }

        variants: List[Tuple[str, str, float]] = []
        logits_by_mode = {}
        for mode in args.decode_modes:
            sequence, pair_max, seq_logits = run_one(module, trans, rotmats, mask, mode, device)
            variants.append((mode, sequence, pair_max))
            logits_by_mode[mode] = seq_logits
        if args.include_native and native is not None:
            variants.append(("native", native, np.nan))

        for mode, sequence, pair_max in variants:
            # The evaluator pairs PDB and FASTA by stem.  Each method therefore
            # gets a private copy of the identical target backbone.
            eval_pdb = target_dir / f"{target_id}__{mode}.pdb"
            shutil.copy2(source, eval_pdb)
            fasta = eval_pdb.with_suffix(".fasta")
            SeqIO.write(
                [SeqRecord(Seq(sequence), id=eval_pdb.stem,
                           description=f"inverse_fold mode={mode}")],
                str(fasta), "fasta",
            )
            eval_paths.append(str(eval_pdb))
            rec, ppl = None, None
            if native is not None and mode in logits_by_mode:
                rec, ppl = sequence_metrics(logits_by_mode[mode], native)
            elif native is not None and mode == "native":
                rec, ppl = 1.0, None
            rows.append({
                "target_id": target_id,
                "pdb_name": eval_pdb.stem,
                "source_pdb": str(source),
                "target_pdb": str(eval_pdb),
                "fasta_path": str(fasta),
                "length": length,
                "length_bin": length_bin(length),
                "decode_mode": mode,
                "evaluation_scope": evaluation_scope,
                "sequence": sequence,
                "pair_prob_max": pair_max,
                "recovery": rec,
                "perplexity": ppl,
            })

    design_df = pd.DataFrame(rows)
    design_df.to_csv(out / "inverse_folding_sequences.csv", index=False)
    if args.skip_rhofold:
        print(f"Saved {len(design_df)} sequences to {out}")
        return

    evaluator = make_evaluator(args, sequence_source="e4")
    eval_dir = out / "external_eval"
    raw_summary = evaluator.evaluate(eval_paths, str(eval_dir))
    with open(out / "external_eval_summary.json", "w", encoding="utf-8") as handle:
        json.dump(raw_summary, handle, indent=2)
    eval_df = standardize_eval(pd.read_csv(find_external_csv(eval_dir)))
    metric_cols = [c for c in (
        "pdb_name", "tm_score", "rmsd", "gdt_ts", "gdt_ha",
        "tm_valid", "rmsd_valid", "joint_valid", "pred_pdb_path"
    ) if c in eval_df.columns]
    merged = design_df.merge(
        eval_df[metric_cols].drop_duplicates("pdb_name", keep="last"),
        on="pdb_name", how="left", validate="one_to_one",
    )
    if "tm_valid" not in merged and "tm_score" in merged:
        merged["tm_valid"] = (merged.tm_score >= args.tm_threshold).astype(float)
    if "rmsd_valid" not in merged and "rmsd" in merged:
        merged["rmsd_valid"] = (merged.rmsd <= args.rmsd_threshold).astype(float)
    if "joint_valid" not in merged and {"tm_valid", "rmsd_valid"} <= set(merged):
        merged["joint_valid"] = (merged.tm_valid.astype(bool) & merged.rmsd_valid.astype(bool)).astype(float)

    merged.to_csv(out / "inverse_folding_with_rhofold.csv", index=False)
    overall = summarize(merged, ["decode_mode"])
    by_length = summarize(merged, ["decode_mode", "length_bin"])
    overall.to_csv(out / "inverse_folding_summary.csv", index=False)
    by_length.to_csv(out / "inverse_folding_by_length.csv", index=False)

    if args.include_grnade:
        print(f"\nRunning gRNAde with n_samples={args.grnade_n_samples} ...")
        grnade_eval_dir = out / "external_eval_grnade"
        grnade_input_dir = out / "grnade_compatible_targets"
        grnade_target_paths = []
        grnade_target_metadata = {}
        for original_path in unique_target_paths:
            original = Path(original_path)
            original_id = original.stem
            meta = target_metadata[original_id]
            native_sequence = meta.get("native_sequence")
            if not native_sequence:
                print(f"[skip gRNAde: missing canonical sequence] {original_id}")
                continue
            compatible = grnade_input_dir / f"{original_id}__grnade_input.pdb"
            write_grnade_compatible_pdb(original, compatible, native_sequence)
            grnade_target_paths.append(str(compatible))
            grnade_target_metadata[compatible.stem] = meta
        grnade_evaluator = make_evaluator(
            args,
            sequence_source="grnade",
            n_sequences=args.grnade_n_samples,
            temperature=args.grnade_temperature,
        )
        grnade_summary = grnade_evaluator.evaluate(grnade_target_paths, str(grnade_eval_dir))
        with open(out / "external_eval_grnade_summary.json", "w", encoding="utf-8") as handle:
            json.dump(grnade_summary, handle, indent=2)
        grnade_best_csv = grnade_eval_dir / "external_eval_best.csv"
        if not grnade_best_csv.exists():
            raise FileNotFoundError(f"Missing gRNAde best-result CSV: {grnade_best_csv}")
        grnade_df = standardize_eval(pd.read_csv(grnade_best_csv))
        grnade_rows = []
        for row in grnade_df.to_dict("records"):
            pdb_name = str(row.get("pdb_name", ""))
            meta = grnade_target_metadata.get(pdb_name)
            if meta is None:
                continue
            grnade_rows.append({
                **meta,
                "pdb_name": pdb_name,
                "fasta_path": row.get("fasta_path", ""),
                "decode_mode": (
                    "grnade" if args.grnade_n_samples == 1
                    else f"grnade_best_of_{args.grnade_n_samples}"
                ),
                "sequence": row.get("sequence", ""),
                "pair_prob_max": np.nan,
                **{key: row.get(key) for key in (
                    "tm_score", "rmsd", "gdt_ts", "gdt_ha", "tm_valid",
                    "rmsd_valid", "joint_valid", "pred_pdb_path",
                    "perplexity", "recovery",
                ) if key in row},
            })
        if grnade_rows:
            merged = pd.concat([merged, pd.DataFrame(grnade_rows)], ignore_index=True, sort=False)

    merged.to_csv(out / "inverse_folding_with_rhofold.csv", index=False)
    overall = summarize(merged, ["decode_mode"])
    by_length = summarize(merged, ["decode_mode", "length_bin"])
    overall.to_csv(out / "inverse_folding_summary.csv", index=False)
    by_length.to_csv(out / "inverse_folding_by_length.csv", index=False)
    print("\n=== Inverse-folding summary ===")
    print(overall.to_string(index=False))
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
