"""PDB readers shared by fixed-backbone decoding and evaluation."""

import numpy as np
import torch
from Bio.PDB import PDBParser

from ribocouple.data import rigid_utils as ru


RNA_RESNAME = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
}


def pdb_frames_and_native_sequence(pdb_path: str):
    """Extract one RNA chain/model using C3'/C4'/O4' frames."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", pdb_path)
    rows = []
    for model in structure:
        # Use the longest RNA-like chain with complete frame atoms.
        chains = []
        for chain in model:
            residues = []
            for residue in chain:
                name = residue.get_resname().strip().upper()
                if all(atom in residue for atom in ("C3'", "C4'", "O4'")):
                    residues.append((residue, RNA_RESNAME.get(name)))
            if residues:
                chains.append(residues)
        if chains:
            rows = max(chains, key=len)
        break
    if not rows:
        raise ValueError("no RNA chain with complete C3'/C4'/O4' frames")

    c3 = torch.as_tensor(np.asarray([r[0]["C3'"].coord for r in rows]), dtype=torch.float32)
    c4 = torch.as_tensor(np.asarray([r[0]["C4'"].coord for r in rows]), dtype=torch.float32)
    o4 = torch.as_tensor(np.asarray([r[0]["O4'"].coord for r in rows]), dtype=torch.float32)
    rigids = ru.Rigid.from_3_points(p_neg_x_axis=o4, origin=c4, p_xy_plane=c3, eps=1e-8)
    flip = torch.eye(3).unsqueeze(0).repeat(len(rows), 1, 1)
    flip[:, 0, 0] = -1
    flip[:, 2, 2] = -1
    rigids = rigids.compose(ru.Rigid(ru.Rotation(rot_mats=flip), None))
    native = "".join(base or "N" for _, base in rows)
    if "N" in native:
        native = None
    return (
        rigids.get_trans().numpy(),
        rigids.get_rots().get_rot_mats().numpy(),
        np.ones(len(rows), dtype=np.float32),
        native,
    )

