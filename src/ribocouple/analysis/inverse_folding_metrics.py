"""Inverse-folding metrics: native-sequence recovery and 2D agreement.

Predicted pairs come from the ViennaRNA MFE fold of the designed sequence.
Two ground-truth tracks, matching the manuscript: a canonical track
(x3dna find_pair by default, or DSSR via --pairs_engine dssr, filtered to
canonical Watson-Crick + wobble pairs; skipped with --pairs_engine none)
and a pure-Python structural H-bond track that needs no external software.
Metrics are computed over all (i, j) residue pairs, i < j: Matthews
correlation coefficient, base-pair precision/recall/F1, and
native-sequence recovery.

Only the ViennaRNA Python bindings are required for the folding-based
metrics; x3dna/DSSR are needed solely for the canonical track, and neither
is bundled.
"""
import json
import math
import os
import re
import subprocess

from Bio.PDB import PDBParser

CANONICAL = {("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")}


def default_find_pair_bin():
    x3dna_root = os.environ.get("X3DNA", "")
    return os.environ.get(
        "X3DNA_BIN",
        os.path.join(x3dna_root, "bin", "find_pair") if x3dna_root else "find_pair",
    )


def pdb_sequence_and_map(pdb_path):
    """Return (sequence, resnum_to_idx) for the first chain with RNA residues."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    for model in struct:
        for chain in model:
            seq = []
            mapping = {}
            idx = 1
            for res in chain:
                if res.id[0] != " ":
                    continue
                rn = res.get_resname().strip()
                if rn not in ("A", "U", "G", "C", "DA", "DT", "DG", "DC"):
                    continue
                if rn.startswith("D"):
                    continue  # skip DNA
                base = rn[0]
                seq.append(base)
                resnum = res.id[1]
                mapping[resnum] = idx
                idx += 1
            if seq:
                return "".join(seq), mapping
    raise ValueError(f"no RNA chain found in {pdb_path}")


def run_find_pair(pdb_path, find_pair_bin=None):
    import tempfile

    bin_path = find_pair_bin or default_find_pair_bin()
    env = os.environ.copy()
    if os.environ.get("X3DNA"):
        env["X3DNA"] = os.environ["X3DNA"]
    # find_pair writes auxiliary files into the cwd; run it in a temp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [bin_path, os.path.abspath(pdb_path)],
            capture_output=True, text=True, env=env, timeout=60, cwd=tmpdir,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"find_pair failed for {pdb_path}: {(result.stderr or '')[-500:]}"
        )
    return result.stdout


def run_dssr_pairs(pdb_path, dssr_bin=None):
    """Canonical pairs from DSSR JSON output; set of (i, j) 1-indexed positions."""
    import tempfile

    bin_path = dssr_bin or os.environ.get("DSSR_BIN", "x3dna-dssr")
    # DSSR writes dssr-* side files into the cwd; run it in a temp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [bin_path, f"-i={os.path.abspath(pdb_path)}", "--json"],
            capture_output=True, text=True, timeout=120, cwd=tmpdir,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"x3dna-dssr failed for {pdb_path}: {(result.stderr or '')[:200]}"
        )
    data = json.loads(result.stdout)
    _, mapping = pdb_sequence_and_map(pdb_path)
    pairs = set()
    for entry in data.get("pairs", []):
        bp = entry.get("bp", "")
        b1, _, b2 = bp.partition("-")
        if (b1, b2) not in CANONICAL:
            continue
        idxs = []
        for nt_key in ("nt1", "nt2"):
            ref = entry.get(nt_key, "")
            m = re.search(r"(-?\d+)", ref.split(".", 1)[-1])
            if not m:
                break
            idxs.append(mapping.get(int(m.group(1))))
        if len(idxs) == 2 and None not in idxs and idxs[0] != idxs[1]:
            pairs.add((min(idxs), max(idxs)))
    return pairs


def parse_x3dna_pairs(text, length):
    """Parse canonical pairs; return set of (i, j) 1-indexed sequence positions."""
    pairs = set()
    for line in text.splitlines():
        if "#" not in line:
            continue
        head, comment = line.split("#", 1)
        toks = head.split()
        if len(toks) < 2:
            continue
        try:
            i = int(toks[0])
            j = int(toks[1])
        except ValueError:
            continue
        m = re.search(r"\[\.\.([AUGC])\]([AUGC])[-*+]+([AUGC])\[\.\.([AUGC])\]", comment)
        if not m:
            continue
        base_i = m.group(2)
        base_j = m.group(3)
        if (base_i, base_j) not in CANONICAL:
            continue
        if not (1 <= i <= length and 1 <= j <= length):
            continue
        if i == j:
            continue
        pairs.add((min(i, j), max(i, j)))
    return pairs


def fold_sequence(seq):
    """Return set of (i, j) pairs from the ViennaRNA MFE dot-bracket."""
    import RNA  # ViennaRNA python bindings

    structure, _ = RNA.fold(seq)
    return dotbracket_to_pairs(structure)


def dotbracket_to_pairs(structure):
    pairs = set()
    stack = []
    for pos, ch in enumerate(structure, start=1):
        if ch == "(":
            stack.append(pos)
        elif ch == ")":
            if stack:
                i = stack.pop()
                pairs.add((i, pos))
    return pairs


_BASE_N = {"N1", "N2", "N3", "N4", "N6", "N7", "N9"}
_BASE_O = {"O2", "O4", "O6"}
_HB_LO, _HB_HI = 2.5, 3.6


def hbond_pairs(pdb_path):
    """Structural H-bond base pairs (pure Python, no external software).

    Counts heavy-atom H-bond candidates between base atoms (N/O) at
    2.5-3.6 Angstrom with at least one N on each end; a residue pair with
    >= 2 candidates is called paired. Captures non-canonical pairs.
    Returns (pairs, length) with 1-based sequence positions, using the
    longest RNA chain. Matches the manuscript's BP-F1_hbond track.
    """
    import numpy as np

    structure = PDBParser(QUIET=True).get_structure("x", pdb_path)
    model = next(structure.get_models())
    best = None
    for chain in model:
        residues = [r for r in chain if "C4'" in r]
        if residues and (best is None or len(residues) > len(best)):
            best = residues
    if best is None:
        raise ValueError(f"no RNA chain found in {pdb_path}")

    length = len(best)
    base_atoms = []
    for res in best:
        atoms = [(a.name.rstrip(), a.coord) for a in res
                 if a.name.rstrip() in _BASE_N | _BASE_O]
        base_atoms.append(atoms)

    pairs = set()
    for i in range(length):
        for j in range(i + 2, length):
            if not base_atoms[i] or not base_atoms[j]:
                continue
            hb = 0
            for n1, c1 in base_atoms[i]:
                for n2, c2 in base_atoms[j]:
                    d = float(np.linalg.norm(c1 - c2))
                    if _HB_LO <= d <= _HB_HI and (n1.startswith("N") or n2.startswith("N")):
                        hb += 1
            if hb >= 2:
                pairs.add((i + 1, j + 1))
    return pairs, length


def pair_confusion(true_pairs, pred_pairs, n):
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    tn = (n * (n - 1) // 2) - tp - fp - fn
    return tp, fp, fn, tn


def mcc_from_confusion(tp, fp, fn, tn):
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / denom


def f1_from_confusion(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def sequence_recovery(native_seq, designed_seq):
    if len(native_seq) != len(designed_seq):
        raise ValueError(
            f"length mismatch: designed {len(designed_seq)} vs native {len(native_seq)}"
        )
    identical = sum(1 for a, b in zip(native_seq, designed_seq) if a == b)
    return identical / len(native_seq)


def f1_track(gt, predicted, empty_gt_recall=None):
    """Precision/recall/F1 over pair sets with the manuscript's frozen
    empty-prediction convention."""
    tp = len(gt & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    if gt:
        recall = tp / len(gt)
    else:
        recall = empty_gt_recall if empty_gt_recall is not None else 0.0
    if isinstance(recall, float) and math.isnan(recall):
        return precision, recall, float("nan")
    value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, value


def evaluate_design(native_pdb, designed_seq, find_pair_bin=None, native_mcc=True,
                    pairs_engine="find_pair", recovery_only=False):
    """Score one designed sequence against its native backbone.

    Always computes native-sequence recovery. With recovery_only=True that
    is the only metric (no external software needed at all).

    Full mode reports two ground-truth tracks:
      - canonical track (pairs_engine 'find_pair', 'dssr'): canonical
        AU/GC/GU pairs -> 2D MCC and BP-F1 (requires x3dna or DSSR);
      - hbond track (always): structural H-bond heuristic pairs, including
        non-canonical pairs -> BP-F1_hbond (pure Python, license-free).
    pairs_engine='none' skips the canonical track (recovery + BP-F1_hbond
    only; needs ViennaRNA but no annotation binary).
    """
    designed_seq = designed_seq.upper().replace("T", "U")
    native_seq, _ = pdb_sequence_and_map(native_pdb)
    length = len(native_seq)
    recovery = sequence_recovery(native_seq, designed_seq)
    if recovery_only:
        return {"length": length, "recovery": recovery}

    pred_pairs = fold_sequence(designed_seq)

    # hbond track: structural truth, no external software
    hb_pairs, hb_length = hbond_pairs(native_pdb)
    empty_recall = None
    if not pred_pairs and not hb_pairs:
        empty_recall = 1.0 if len(designed_seq) == hb_length else float("nan")
    p_hb, r_hb, f_hb = f1_track(hb_pairs, pred_pairs, empty_gt_recall=empty_recall)

    row = {
        "length": length,
        "recovery": recovery,
        "n_pred_pairs": len(pred_pairs),
        "n_hbond_pairs": len(hb_pairs),
        "bp_hbond_precision": p_hb,
        "bp_hbond_recall": r_hb,
        "bp_hbond_f1": f_hb,
    }

    if pairs_engine != "none":
        if pairs_engine == "dssr":
            true_pairs = run_dssr_pairs(native_pdb, dssr_bin=find_pair_bin)
        else:
            true_pairs = parse_x3dna_pairs(run_find_pair(native_pdb, find_pair_bin), length)
        tp, fp, fn, tn = pair_confusion(true_pairs, pred_pairs, length)
        precision, recall, f1 = f1_from_confusion(tp, fp, fn)
        row.update({
            "n_true_pairs": len(true_pairs),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "mcc": mcc_from_confusion(tp, fp, fn, tn),
            "bp_precision": precision,
            "bp_recall": recall,
            "bp_f1": f1,
        })
        if native_mcc:
            nv_tp, nv_fp, nv_fn, nv_tn = pair_confusion(
                true_pairs, fold_sequence(native_seq), length
            )
            row["native_mcc"] = mcc_from_confusion(nv_tp, nv_fp, nv_fn, nv_tn)
    return row
