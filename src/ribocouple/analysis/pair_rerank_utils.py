"""
Utilities for pair-aware reranking and pair consistency evaluation.

E2.5:  Given N generated backbones, rank them by topology(pair) confidence
       and select top-K for external evaluation (gRNAde + RhoFold + TMscore).

E2.5+: Given a generated backbone and its RhoFold-folded structure,
       compare the model's predicted base pairs to the base pairs inferred
       from the folded structure, checking WC complementarity.
"""

import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBIO import PDBIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from ribocouple.data import utils as du
from ribocouple.data import all_atom as rna_all_atom
from ribocouple.data.interpolant import Interpolant
from ribocouple.models.flow_module import FlowModule


# ---------------------------------------------------------------------------
# Model / sampling helpers
# ---------------------------------------------------------------------------

def load_flow_module(ckpt_path: str, cfg=None):
    """
    Load a FlowModule from a PyTorch Lightning checkpoint.

    Args:
        ckpt_path: path to .ckpt file.
        cfg: optional OmegaConf config. If not provided, tries to recover from
             checkpoint['hyper_parameters'].

    Returns:
        FlowModule on CPU in eval mode.
    """
    from omegaconf import OmegaConf
    from datetime import datetime

    # Register 'now' resolver globally (OmegaConf < 2.3 needs explicit registration)
    try:
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    except Exception:
        pass

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if cfg is None:
        if "hyper_parameters" in checkpoint and "cfg" in checkpoint["hyper_parameters"]:
            cfg = checkpoint["hyper_parameters"]["cfg"]
        else:
            raise ValueError(
                "Config not found in checkpoint. Please provide --config_path."
            )

    # Resolve ${now:...} forcefully: set target key directly (avoids resolver issues)
    try:
        cfg.experiment.checkpointer.dirpath = "/tmp/checkpoints"
    except Exception:
        pass
    try:
        OmegaConf.resolve(cfg)
    except Exception:
        pass

    model = FlowModule(cfg)
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:10])
        unexpected = ", ".join(incompatible.unexpected_keys[:10])
        problems = []
        if incompatible.missing_keys:
            problems.append(
                f"missing_keys={len(incompatible.missing_keys)} [{missing}]"
            )
        if incompatible.unexpected_keys:
            problems.append(
                f"unexpected_keys={len(incompatible.unexpected_keys)} [{unexpected}]"
            )
        raise RuntimeError(
            "Checkpoint is incompatible with the current FlowModule. "
            "Pair reranking requires a fully compatible checkpoint with pair-head "
            f"weights loaded. Details: {'; '.join(problems)}"
        )
    model.eval()
    return model

def generate_samples(model, cfg, lengths: List[int], n_samples_per_length: int,
                     device: str = 'cuda', seed: int = 123, return_sequences: bool = False):
    """
    Generate unconditional backbone samples for a list of lengths.

    Args:
        model: FlowModule in eval mode.
        cfg: model + interpolant config.
        lengths: list of sequence lengths.
        n_samples_per_length: number of independent samples per length.
        device: torch device.
        seed: random seed.
        return_sequences: if True, also return E4-predicted sequence indices.

    Returns:
        List of dicts with keys:
            - length (int)
            - sample_idx (int)
            - atom37 (np.ndarray, [L, 37, 3])
            - res_mask (np.ndarray, [L])
            - seq_indices (np.ndarray, optional) [L]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    interpolant = Interpolant(cfg.interpolant)
    interpolant.set_device(device)

    samples = []
    for length in lengths:
        print(f"Generating {n_samples_per_length} samples of length {length} ...")
        if return_sequences:
            atom37_traj, _, _, model_out = interpolant.sample(
                n_samples_per_length, length, model.model, return_model_out=True
            )
            seq_logits = model_out.get('seq_logits', None)
            if seq_logits is not None:
                seq_indices = torch.argmax(seq_logits, dim=-1).cpu().numpy()
            else:
                seq_indices = None
        else:
            atom37_traj, _, _ = interpolant.sample(n_samples_per_length, length, model.model)
            seq_indices = None
        final_atom37 = atom37_traj[-1].numpy()  # [B, L, 37, 3]

        for b in range(n_samples_per_length):
            sample = {
                'length': length,
                'sample_idx': b,
                'atom37': final_atom37[b],
                'res_mask': np.ones(length, dtype=bool),
            }
            if seq_indices is not None:
                sample['seq_indices'] = seq_indices[b]
            samples.append(sample)
    return samples


def atom37_to_trans_rot(atom37: torch.Tensor):
    """
    Reconstruct backbone frames (trans, rotmats) from ATOM37 coordinates.

    Uses the same convention as the training data pipeline:
      - origin = C4'  (atom index 3)
      - p_neg_x_axis = O4' (atom index 6)
      - p_xy_plane = C3' (atom index 2)
      - followed by the same axis-flip rotation used in data_transforms.py

    Args:
        atom37: [..., L, 37, 3] torch.Tensor.

    Returns:
        trans: [..., L, 3]
        rotmats: [..., L, 3, 3]
    """
    from ribocouple.data import rigid_utils as ru

    # Frame atoms: [O4', C4', C3'] -> indices [6, 3, 2]
    p_neg_x_axis = atom37[..., 6, :]
    origin = atom37[..., 3, :]
    p_xy_plane = atom37[..., 2, :]

    rigids = ru.Rigid.from_3_points(
        p_neg_x_axis=p_neg_x_axis,
        origin=origin,
        p_xy_plane=p_xy_plane,
        eps=1e-8,
    )

    # Same axis-flip rotation as data_transforms.py
    rot_dim = atom37.shape[:-3]
    rots = torch.eye(3, dtype=atom37.dtype, device=atom37.device)
    rots = rots.view(*((1,) * len(rot_dim)), 1, 3, 3).expand(*rot_dim, -1, -1, -1).clone()
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    flip_rigid = ru.Rigid(ru.Rotation(rot_mats=rots), None)
    rigids = rigids.compose(flip_rigid)

    return rigids.get_trans(), rigids.get_rots().get_rot_mats()


def run_model_on_atom37(model, atom37: np.ndarray, res_mask: np.ndarray,
                        t: float = 1.0, device: str = 'cuda'):
    """
    Run a forward pass on a clean/generated backbone to get pair_logits.

    Args:
        model: FlowModule.
        atom37: [L, 37, 3] numpy array.
        res_mask: [L] bool array.
        t: diffusion timestep to use. Default 1.0 (final step).
        device: torch device.

    Returns:
        dict with "pair_logits" ([L, L] torch.Tensor on CPU),
        "pair_type_logits" and "seq_logits" (each [L, L, ...] tensor or None).
    """
    model = model.to(device)
    atom37_t = torch.from_numpy(atom37).float().unsqueeze(0).to(device)
    res_mask_t = torch.from_numpy(res_mask).float().unsqueeze(0).to(device)

    trans, rotmats = atom37_to_trans_rot(atom37_t)

    batch = {
        'res_mask': res_mask_t,
        'trans_t': trans,
        'rotmats_t': rotmats,
        't': torch.full((1, 1), t, device=device),
    }

    with torch.no_grad():
        out = model.model(batch)

    #return out['pair_logits'].squeeze(0).cpu()
    return {
         "pair_logits": out["pair_logits"].squeeze(0).cpu(),
         "pair_type_logits": (
               out["pair_type_logits"].squeeze(0).cpu()
               if out.get("pair_type_logits") is not None
               else None
           ),
         "seq_logits": (
             out["seq_logits"].squeeze(0).cpu()
             if out.get("seq_logits") is not None
             else None
          ),
     }

# ---------------------------------------------------------------------------
# Pair score computation for reranking
# ---------------------------------------------------------------------------

def compute_pair_scores(pair_logits: torch.Tensor, res_mask: np.ndarray,
                        topk: int = 10) -> Dict[str, float]:
    """
    Compute various pair-based confidence scores from pair_logits.

    Args:
        pair_logits: [L, L] logits.
        res_mask: [L] bool mask.
        topk: number of top pair probabilities to sum.

    Returns:
        dict of scalar scores.
    """
    L = pair_logits.shape[0]
    # Directed upper-triangle statistic (manuscript: sum of the ten largest
    # binary pair probabilities, directed entry i<j per unordered pair).
    pair_probs = torch.sigmoid(pair_logits)

    device = pair_logits.device
    eye = torch.eye(L, device=device).bool()
    upper = torch.triu(torch.ones(L, L, device=device).bool(), diagonal=1)
    mask_t = torch.from_numpy(res_mask).to(device)
    pair_mask = mask_t[:, None] & mask_t[None, :] & upper & ~eye

    scores = {}

    # Mean probability over valid upper-triangle positions
    scores['pair_prob_mean'] = (
        (pair_probs * pair_mask).sum() / (pair_mask.sum() + 1e-8)
    ).item()

    # Max probability
    scores['pair_prob_max'] = (
        pair_probs.masked_fill(~pair_mask, 0.0).max()
    ).item()

    # Number of predicted pairs at threshold 0.5
    scores['n_pairs_pred'] = (
        ((pair_probs > 0.5) & pair_mask).sum()
    ).item()

    # Sum of top-K probabilities
    probs_masked = pair_probs.masked_fill(~pair_mask, 0.0)
    k = min(topk, int(pair_mask.sum().item()))
    if k > 0:
        flat = probs_masked.view(-1)
        topk_sum = flat.topk(k).values.sum()
        scores['topk_pair_sum'] = topk_sum.item()
    else:
        scores['topk_pair_sum'] = 0.0

    # Entropy of the (masked) pair distribution
    # Lower entropy -> model is more certain about topology.
    probs_flat = probs_masked.view(-1)
    probs_flat = probs_flat / (probs_flat.sum() + 1e-8)
    entropy = -(probs_flat * torch.log(probs_flat + 1e-8)).sum()
    scores['pair_entropy'] = entropy.item()

    # Standard deviation over masked positions
    masked_vals = pair_probs[pair_mask]
    scores["pair_prob_std"] = masked_vals.std().item() if masked_vals.numel() > 1 else 0.0

    return scores


# ---------------------------------------------------------------------------
# Base pair extraction from PDB
# ---------------------------------------------------------------------------

# Heavy atoms in each nucleotide base (excluding ribose phosphate)
BASE_ATOMS = {
    'A': ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9'],
    'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
    'G': ['N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9'],
    'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
}

WC_PAIRS = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}


def extract_sequence_and_base_atoms(pdb_path: str):
    """
    Extract RNA sequence and base/C1' atom coordinates from a PDB file.

    Returns:
        seq (str): RNA sequence, '?' for unknown residues.
        c1p_coords (np.ndarray): [L, 3]
        base_atom_coords (List[List[Tuple[str, np.ndarray]]]): per residue,
            list of (atom_name, coord).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rna', pdb_path)

    seq_chars = []
    c1p_coords = []
    base_atom_coords = []

    # Take first model, first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())

    for residue in chain.get_residues():
        resname = residue.get_resname().strip()
        # Standard RNA residue names in PDB
        resname_to_letter = {
            'A': 'A', 'ADE': 'A',
            'C': 'C', 'CYT': 'C',
            'G': 'G', 'GUA': 'G',
            'U': 'U', 'URA': 'U',
        }
        letter = resname_to_letter.get(resname, '?')
        seq_chars.append(letter)

        # C1' atom
        c1p = None
        if "C1'" in residue:
            c1p = residue["C1'"].get_coord()
        c1p_coords.append(c1p)

        # Base heavy atoms
        base_atoms = []
        if letter in BASE_ATOMS:
            for atom_name in BASE_ATOMS[letter]:
                if atom_name in residue:
                    base_atoms.append((atom_name, residue[atom_name].get_coord()))
        base_atom_coords.append(base_atoms)

    seq = ''.join(seq_chars)
    c1p_coords = np.array([c if c is not None else [np.nan] * 3 for c in c1p_coords])
    return seq, c1p_coords, base_atom_coords


def extract_base_pairs_from_pdb(pdb_path: str,
                                c1c1_min: float = 8.5,
                                c1c1_max: float = 13.0,
                                hbond_cutoff: float = 3.8,
                                require_wc: bool = False):
    """
    Extract base pairs from a PDB structure using simple geometric criteria.

    A pair (i, j) is called if:
      1. C1'-C1' distance is within [c1c1_min, c1c1_max] Angstroms.
      2. At least one heavy atom in base i is within hbond_cutoff of a heavy
         atom in base j.
      3. If require_wc=True, only keep A-U / G-C / G-U pairs.

    Returns:
        List of dicts with keys: i, j, seq_i, seq_j, c1c1_dist, min_dist, is_wc.
    """
    seq, c1p_coords, base_atom_coords = extract_sequence_and_base_atoms(pdb_path)
    L = len(seq)
    pairs = []

    for i in range(L):
        for j in range(i + 1, L):
            si, sj = seq[i], seq[j]
            if si == '?' or sj == '?':
                continue

            # C1'-C1' distance filter
            c1i, c1j = c1p_coords[i], c1p_coords[j]
            if np.isnan(c1i).any() or np.isnan(c1j).any():
                continue
            c1c1_dist = np.linalg.norm(c1i - c1j)
            if not (c1c1_min <= c1c1_dist <= c1c1_max):
                continue

            # Heavy atom distance filter
            atoms_i = base_atom_coords[i]
            atoms_j = base_atom_coords[j]
            if len(atoms_i) == 0 or len(atoms_j) == 0:
                continue

            min_dist = float('inf')
            for _, ci in atoms_i:
                for _, cj in atoms_j:
                    d = np.linalg.norm(ci - cj)
                    if d < min_dist:
                        min_dist = d

            if min_dist > hbond_cutoff:
                continue

            is_wc = (si, sj) in WC_PAIRS
            if require_wc and not is_wc:
                continue

            pairs.append({
                'i': i,
                'j': j,
                'seq_i': si,
                'seq_j': sj,
                'c1c1_dist': float(c1c1_dist),
                'min_dist': float(min_dist),
                'is_wc': is_wc,
            })

    return seq, pairs


# ---------------------------------------------------------------------------
# Pair consistency
# ---------------------------------------------------------------------------

def get_predicted_pairs(pair_probs: np.ndarray, res_mask: np.ndarray,
                        threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Get predicted pair list from model pair probabilities.

    Returns:
        List of (i, j, prob) for upper-triangle positions with prob > threshold.
    """
    L = pair_probs.shape[0]
    pairs = []
    for i in range(L):
        if not res_mask[i]:
            continue
        for j in range(i + 1, L):
            if not res_mask[j]:
                continue
            if pair_probs[i, j] > threshold:
                pairs.append((i, j, float(pair_probs[i, j])))
    return pairs


def compute_pair_consistency(pred_pairs: List[Tuple[int, int, float]],
                             struct_pairs: List[Dict],
                             seq: str,
                             res_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compare model predicted pairs to structural pairs from RhoFold.

    Metrics:
      - pair_consistency: fraction of predicted pairs that are both
          structurally present AND sequence-consistent (WC).
      - struct_recall: fraction of structural pairs that are also predicted.
      - wc_consistency: fraction of predicted pairs that are WC pairs.
      - precision_vs_struct: predicted pairs that are structural / all predicted.
      - recall_vs_struct: structural pairs that are predicted / all structural.

    Args:
        pred_pairs: list of (i, j, prob) from model.
        struct_pairs: list of dicts from extract_base_pairs_from_pdb.
        seq: RNA sequence from folded structure.
        res_mask: optional residue mask.

    Returns:
        Dict of metrics.
    """
    pred_set = {(i, j) for i, j, _ in pred_pairs}
    struct_set = {(p['i'], p['j']) for p in struct_pairs}
    struct_wc_set = {(p['i'], p['j']) for p in struct_pairs if p['is_wc']}

    # Predicted pairs that are also found in structure
    pred_and_struct = pred_set & struct_set
    # Predicted pairs that are structural AND WC
    pred_and_struct_wc = pred_set & struct_wc_set

    # Predicted pairs whose sequence composition is WC
    wc_pred = set()
    for i, j, _ in pred_pairs:
        if (seq[i], seq[j]) in WC_PAIRS:
            wc_pred.add((i, j))

    metrics = {}

    metrics['n_pred_pairs'] = len(pred_set)
    metrics['n_struct_pairs'] = len(struct_set)
    metrics['n_struct_wc_pairs'] = len(struct_wc_set)

    # Key metric: predicted pairs that are structural + WC
    metrics['pair_consistency'] = (
        len(pred_and_struct_wc) / len(pred_set) if pred_set else 0.0
    )

    # Predicted pairs that are structural (regardless of WC)
    metrics['precision_vs_struct'] = (
        len(pred_and_struct) / len(pred_set) if pred_set else 0.0
    )

    # Structural WC pairs recovered by model
    metrics['recall_vs_struct_wc'] = (
        len(pred_and_struct_wc) / len(struct_wc_set) if struct_wc_set else 0.0
    )

    # All structural pairs recovered by model
    metrics['recall_vs_struct'] = (
        len(pred_and_struct) / len(struct_set) if struct_set else 0.0
    )

    # Fraction of predicted pairs that are WC by sequence
    metrics['wc_consistency'] = (
        len(wc_pred & pred_set) / len(pred_set) if pred_set else 0.0
    )

    return metrics


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_atom37_to_pdb(atom37: np.ndarray, output_path: str):
    """Write atom37 array to PDB using existing project util."""
    from ribocouple.analysis import utils as au
    return au.write_complex_to_pdbs(
        atom37,
        output_path,
        is_na_residue_mask=np.ones(atom37.shape[0], dtype=bool),
    )


def save_pair_predictions_csv(pair_records: List[Dict], csv_path: str):
    """Save sparse pair predictions to CSV."""
    if not pair_records:
        # Write empty with header
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'length', 'sample_idx', 'pdb_path', 'i', 'j', 'prob'
            ])
            writer.writeheader()
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'length', 'sample_idx', 'pdb_path', 'i', 'j', 'prob'
        ])
        writer.writeheader()
        writer.writerows(pair_records)


def load_pair_predictions_csv(csv_path: str) -> Dict[Tuple[int, int], List[Tuple[int, int, float]]]:
    """Load sparse pair predictions indexed by (length, sample_idx)."""
    df = pd.read_csv(csv_path)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[(int(row['length']), int(row['sample_idx']))].append(
            (int(row['i']), int(row['j']), float(row['prob']))
        )
    return groups


# ---------------------------------------------------------------------------
# Additional helpers for extended analysis
# ---------------------------------------------------------------------------

def compute_structural_consistency(pred_pairs: List[Tuple[int, int, float]],
                                   struct_pairs: List[Dict]) -> Dict[str, float]:
    """
    Geometric-only consistency: compare predicted pairs to structural pairs
    without checking WC complementarity. Useful for generated PDBs whose
    residue names are placeholders.

    Returns:
        precision: fraction of predicted pairs found in structure
        recall: fraction of structural pairs also predicted
        f1: harmonic mean
        n_pred_pairs, n_struct_pairs
    """
    pred_set = {(i, j) for i, j, _ in pred_pairs}
    struct_set = {(p['i'], p['j']) for p in struct_pairs}

    pred_and_struct = pred_set & struct_set

    precision = len(pred_and_struct) / len(pred_set) if pred_set else 0.0
    recall = len(pred_and_struct) / len(struct_set) if struct_set else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'structural_precision': precision,
        'structural_recall': recall,
        'structural_f1': f1,
        'n_pred_pairs': len(pred_set),
        'n_struct_pairs': len(struct_set),
    }


def compute_pair_score_correlations(df: pd.DataFrame,
                                    score_cols: List[str],
                                    metric_cols: List[str]) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between pair scores and structure metrics.

    Args:
        df: DataFrame with pair scores and structure metrics per sample.
        score_cols: list of pair score column names.
        metric_cols: list of structure metric column names (e.g. tm_score, rmsd, joint_valid).

    Returns:
        DataFrame with rows = score_cols, columns = metric_cols, values = Pearson r.
    """
    try:
        from scipy.stats import pearsonr, spearmanr
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    def _pearson(x, y):
        if HAS_SCIPY:
            return pearsonr(x, y)[0]
        # Fallback to numpy
        return np.corrcoef(x, y)[0, 1]

    def _spearman(x, y):
        if HAS_SCIPY:
            return spearmanr(x, y)[0]
        # Fallback: rank-based Pearson using numpy argsort
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return np.corrcoef(rx, ry)[0, 1]

    rows = []
    for score_col in score_cols:
        row = {'score': score_col}
        for metric_col in metric_cols:
            if score_col not in df.columns or metric_col not in df.columns:
                row[f'{metric_col}_pearson'] = float('nan')
                row[f'{metric_col}_spearman'] = float('nan')
                continue

            valid = df[[score_col, metric_col]].dropna()
            if len(valid) < 3:
                row[f'{metric_col}_pearson'] = float('nan')
                row[f'{metric_col}_spearman'] = float('nan')
                continue

            x = valid[score_col].values
            y = valid[metric_col].values

            try:
                r_pearson = _pearson(x, y)
            except Exception:
                r_pearson = float('nan')

            try:
                r_spearman = _spearman(x, y)
            except Exception:
                r_spearman = float('nan')

            row[f'{metric_col}_pearson'] = r_pearson
            row[f'{metric_col}_spearman'] = r_spearman
        rows.append(row)

    return pd.DataFrame(rows)


def load_per_sample_metrics(csv_path: str) -> pd.DataFrame:
    """Load per-sample structure metrics CSV."""
    df = pd.read_csv(csv_path)
    required = {'pdb_path', 'tm_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Got: {list(df.columns)}")
    return df


def find_generated_pdb_path(pdb_name_or_path: str, search_dirs: List[str]) -> Optional[str]:
    """
    Try to resolve a PDB name or relative path to an absolute path by searching
    in multiple directories.
    """
    if os.path.isfile(pdb_name_or_path):
        return os.path.abspath(pdb_name_or_path)

    basename = os.path.basename(pdb_name_or_path)
    for d in search_dirs:
        candidate = os.path.join(d, basename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        # Try without .pdb extension if missing
        if not basename.endswith('.pdb'):
            candidate = os.path.join(d, basename + '.pdb')
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    return None


def pdb_to_trans_rot(pdb_path: str):
    """
    Read a PDB file and reconstruct backbone frames (trans, rotmats) using the
    same convention as atom37_to_trans_rot:
      - origin = C4'
      - p_neg_x_axis = O4'
      - p_xy_plane = C3'

    Returns:
        trans: [L, 3] np.ndarray
        rotmats: [L, 3, 3] np.ndarray
        res_mask: [L] bool np.ndarray
    """
    from Bio.PDB import PDBParser
    from ribocouple.data import rigid_utils as ru

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rna', pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())

    c3p, c4p, o4p = [], [], []
    for residue in chain.get_residues():
        coords = {}
        for atom_name in ["C3'", "C4'", "O4'"]:
            if atom_name in residue:
                coords[atom_name] = residue[atom_name].get_coord()
        if len(coords) == 3:
            c3p.append(coords["C3'"])
            c4p.append(coords["C4'"])
            o4p.append(coords["O4'"])

    if len(c4p) == 0:
        raise ValueError(f"No C4' atoms found in {pdb_path}")

    c3p = torch.from_numpy(np.array(c3p, dtype=np.float32))
    c4p = torch.from_numpy(np.array(c4p, dtype=np.float32))
    o4p = torch.from_numpy(np.array(o4p, dtype=np.float32))

    rigids = ru.Rigid.from_3_points(
        p_neg_x_axis=o4p,
        origin=c4p,
        p_xy_plane=c3p,
        eps=1e-8,
    )

    L = len(c4p)
    rots = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(L, -1, -1).clone()
    rots[:, 0, 0] = -1
    rots[:, 2, 2] = -1
    flip_rigid = ru.Rigid(ru.Rotation(rot_mats=rots), None)
    rigids = rigids.compose(flip_rigid)

    trans = rigids.get_trans().numpy()
    rotmats = rigids.get_rots().get_rot_mats().numpy()
    res_mask = np.ones(L, dtype=bool)

    return trans, rotmats, res_mask


def run_model_on_pdb(model, pdb_path: str, t: float = 1.0, device: str = 'cuda'):
    """
    Run a forward pass on a PDB file to get pair_logits.

    Args:
        model: FlowModule.
        pdb_path: path to PDB file.
        t: diffusion timestep.
        device: torch device.

    Returns:
        pair_logits: [L, L] torch.Tensor on CPU.
    """
    model = model.to(device)
    trans, rotmats, res_mask = pdb_to_trans_rot(pdb_path)

    trans_t = torch.from_numpy(trans).float().unsqueeze(0).to(device)
    rotmats_t = torch.from_numpy(rotmats).float().unsqueeze(0).to(device)
    res_mask_t = torch.from_numpy(res_mask).float().unsqueeze(0).to(device)

    batch = {
        'res_mask': res_mask_t,
        'trans_t': trans_t,
        'rotmats_t': rotmats_t,
        't': torch.full((1, 1), t, device=device),
    }

    with torch.no_grad():
        out = model.model(batch)

    return out['pair_logits'].squeeze(0).cpu()
