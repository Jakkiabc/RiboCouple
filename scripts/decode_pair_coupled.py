#!/usr/bin/env python3
"""decode_pair_coupled.py — pair-coupled decoding with a frozen pair-type model.

Default rule (matches the manuscript): directed pair probabilities are
symmetrized by their arithmetic mean; candidate edges are all residue pairs
i<j whose symmetrized probability is at least 0.5; edges are added greedily
in decreasing order of probability, skipping any edge whose two residues are
already used, so the retained set is non-overlapping. Each retained edge is
jointly reassigned to the highest-scoring member of {AU, UA, GC, CG, GU, UG},
maximizing the sum of the two log-softmax nucleotide probabilities and the
corresponding pair-type logit. Unselected residues keep their argmax
assignments.

An alternative mutual-best variant (probability threshold 0.8, minimum
sequence separation 4) is available via --rule mutual. --rule argmax skips
pair selection entirely and returns plain argmax decoding (the "RiboCouple
argmax" baseline in the manuscript).

Usage:
  python decode_pair_coupled.py --ckpt rb_pairtype_e198.ckpt --pdb target.pdb [--out seq.fasta]

Cleaned release entrypoint (paths by arguments, no hard-coded host paths).
"""
import argparse, sys
import torch

from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from ribocouple.data.pdb_io import pdb_frames_and_native_sequence
from ribocouple.models.checkpoint import load_flow_module

BASES = ["A", "C", "G", "U"]
# pair-type classes: 0 = AU/UA, 1 = CG/GC, 2 = GU/UG (A=0, C=1, G=2, U=3)
CANON = {0: ((0, 3), (3, 0)), 1: ((1, 2), (2, 1)), 2: ((2, 3), (3, 2))}


def select_greedy_pairs(pair_probs, threshold=0.5):
    length = pair_probs.shape[0]
    candidates = []
    for i in range(length):
        for j in range(i + 1, length):
            w = float(pair_probs[i, j])
            if w >= threshold:
                candidates.append((w, i, j))
    candidates.sort(reverse=True)
    used = [False] * length
    pairs = []
    for w, i, j in candidates:
        if not used[i] and not used[j]:
            used[i] = used[j] = True
            pairs.append((i, j))
    return pairs


def select_mutual_best_pairs(pair_probs, threshold=0.8, min_sequence_separation=4):
    length = pair_probs.shape[0]
    positions = torch.arange(length)
    sep = torch.abs(positions[:, None] - positions[None, :])
    valid = (sep >= min_sequence_separation).to(pair_probs.device)
    scores = pair_probs.masked_fill(~valid, float("-inf"))
    best = scores.argmax(dim=-1)
    selected = []
    for i in range(length):
        j = int(best[i].item())
        if i >= j or best[j].item() != i:
            continue
        if float(pair_probs[i, j].item()) < threshold:
            continue
        selected.append((i, j))
    return selected


def decode(seq_logits, pairs, pair_type_logits=None):
    log_probs = torch.log_softmax(seq_logits, dim=-1)
    decoded = torch.argmax(seq_logits, dim=-1).clone()
    for i, j in pairs:
        best = None
        for t, variants in CANON.items():
            for a, b in variants:
                s = float(log_probs[i, a]) + float(log_probs[j, b])
                if pair_type_logits is not None:
                    s += float(pair_type_logits[i, j, t])
                if best is None or s > best[0]:
                    best = (s, a, b)
        decoded[i], decoded[j] = best[1], best[2]
    return "".join(BASES[int(k)] for k in decoded.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rule", choices=["greedy", "mutual", "argmax"], default="greedy",
                    help="edge selection rule; 'greedy' (default) matches the manuscript, "
                         "'argmax' disables pair-coupled reassignment")
    ap.add_argument("--pair_threshold", type=float, default=None,
                    help="default: 0.5 for --rule greedy, 0.8 for --rule mutual")
    ap.add_argument("--min_sep", type=int, default=4,
                    help="minimum sequence separation (only used with --rule mutual)")
    ap.add_argument("--out", default="decode_pair_coupled.fasta")
    args = ap.parse_args()

    if args.pair_threshold is None:
        args.pair_threshold = 0.8 if args.rule == "mutual" else 0.5

    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = load_flow_module(args.ckpt, cfg=None).to(dev).eval()
    trans, rot, mask, _ = pdb_frames_and_native_sequence(args.pdb)
    batch = {
        "res_mask": torch.from_numpy(mask).float().unsqueeze(0).to(dev),
        "trans_t": torch.from_numpy(trans).float().unsqueeze(0).to(dev),
        "rotmats_t": torch.from_numpy(rot).float().unsqueeze(0).to(dev),
        "t": torch.ones((1, 1), device=dev),
    }
    with torch.inference_mode():
        out = model.model(batch)
    pl = out["pair_logits"][0].detach()
    P = torch.sigmoid(pl)
    P = (P + P.T) / 2.0
    M = torch.from_numpy(mask).float().to(dev)
    P = P * (M[:, None] * M[None, :])
    P.fill_diagonal_(0)
    if args.rule == "argmax":
        pairs = []
    elif args.rule == "greedy":
        pairs = select_greedy_pairs(P, threshold=args.pair_threshold)
    else:
        pairs = select_mutual_best_pairs(P, threshold=args.pair_threshold,
                                         min_sequence_separation=args.min_sep)
    tl = out.get("pair_type_logits")
    tl = tl[0].detach() if tl is not None else None
    seq = decode(out["seq_logits"][0].detach(), pairs, tl)
    header = f">pair_coupled rule={args.rule} n_pairs={len(pairs)} threshold={args.pair_threshold}"
    if args.rule == "mutual":
        header += f" min_sep={args.min_sep}"
    with open(args.out, "w") as f:
        f.write(f"{header}\n{seq}\n")
    print(seq)


if __name__ == "__main__":
    main()
