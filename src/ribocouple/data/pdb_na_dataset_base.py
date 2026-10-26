# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import functools as fn
import os
import numpy as np
import pandas as pd

import torch, tree
from beartype.typing import Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ribocouple.data import data_transforms
from ribocouple.data import utils as du
from ribocouple.data import rigid_utils

NUM_NA_RESIDUE_ATOMS = 23

class PDBNABaseDataset(Dataset):
    def __init__(
        self,
        data_conf,
        is_training: bool,
        filter_eval_split: bool = False,
        inference_cfg: Optional[DictConfig] = None,
    ):
        self._data_conf = data_conf
        self._is_training = is_training
        self._filter_eval_split = filter_eval_split
        self._inference_cfg = inference_cfg
        self._init_metadata_and_splits()

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def ddpm(self):
        return self._ddpm

    @property
    def is_training(self):
        return self._is_training

    @property
    def filter_eval_split(self):
        return self._filter_eval_split

    @property
    def inference_cfg(self):
        return self._inference_cfg

    @property
    def sequence_ddpm(self):
        return self._sequence_ddpm

    def _init_metadata_and_splits(self): # new
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv # original CSV before filtering and transformations

        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )

        filter_conf = self.data_conf.filtering

        # length-based filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filter_conf.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filter_conf.min_len]

        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"] # ignore multimers
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0] # remove proteins
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)
        # pdb_csv.reset_index(inplace=True) # reset the index to ensure samples are taken within proper bounds

        if self.is_training:
            self.csv = pdb_csv
            print (f"Training: {len(self.csv)} filtered examples") # FINAL post-filtered set for RNASolo from metadata CSV
        else:
            # validation data
            eval_csv = pdb_csv
            explicit_min_len = int(getattr(filter_conf, "min_len", 0))
            explicit_max_len = int(getattr(filter_conf, "max_len", 0))
            explicit_len_step = int(getattr(self._data_conf, "eval_length_step", 10))
            if explicit_len_step <= 0:
                raise ValueError(f"eval_length_step must be positive, got {explicit_len_step}")

            explicit_lengths = np.arange(
                explicit_min_len,
                explicit_max_len + 1,
                explicit_len_step,
            )
            available_lengths = set(eval_csv["modeled_na_seq_len"].unique().tolist())
            eval_lengths = np.array([length for length in explicit_lengths if length in available_lengths])

            if len(eval_lengths) == 0:
                raise ValueError(
                    f"No validation examples found for explicit length range "
                    f"[{explicit_min_len}, {explicit_max_len}] with step {explicit_len_step} after filtering."
                )

            eval_csv = eval_csv[eval_csv.modeled_na_seq_len.isin(eval_lengths)]
            eval_csv = eval_csv.groupby("modeled_na_seq_len").sample(
                self._data_conf.samples_per_eval_length,
                replace=True,
                random_state=123,
            )
            eval_csv = eval_csv.sort_values(["modeled_na_seq_len"], ascending=False)
            self.csv = eval_csv
            print(
                f"Validation: {len(self.csv)} examples with explicit lengths {eval_lengths} "
                f"(step={explicit_len_step})."
            )

    # cache make the same sample in same batch
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_complex_feats(processed_feats)

        # Designate which residues to diffuse and which to fix. By default, diffuse all residues
        diffused_mask = np.ones_like(processed_feats["bb_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        processed_feats["fixed_mask"] = fixed_mask

        # Distinguish between protein residues and nucleic acid residues using corresponding masks
        processed_feats["is_protein_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 0] == 1
        )
        processed_feats["is_na_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 1] == 1
        ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
        na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

        # Find interfaces
        inter_chain_interacting_residue_mask = torch.zeros(len(diffused_mask), dtype=torch.bool)
        inter_chain_interacting_residue_mask[processed_feats["inter_chain_interacting_idx"]] = True

        # Only take modeled residues
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        
        del processed_feats["modeled_idx"]
       
        if processed_feats["protein_modeled_idx"] is None:
            del processed_feats["protein_modeled_idx"]
       
        if processed_feats["na_modeled_idx"] is None:
            del processed_feats["na_modeled_idx"]
       
        processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)
        inter_chain_interacting_residue_mask = inter_chain_interacting_residue_mask[
            min_idx : (max_idx + 1)
        ]

        # Run through OpenFold data transforms.
        chain_feats, na_chain_feats = (
            {
                "aatype": torch.tensor(processed_feats["aatype"]).long(),
                "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
                "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
                "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
            },
            {},
        )

        if na_inputs_present:
            na_chain_feats = {
                "aatype": chain_feats["aatype"][processed_feats["is_na_residue_mask"]],
                "all_atom_positions": chain_feats["all_atom_positions"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "all_atom_mask": chain_feats["all_atom_mask"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "atom_deoxy": chain_feats["atom_deoxy"][processed_feats["is_na_residue_mask"]],
            }
            na_chain_feats["atom23_gt_positions"] = na_chain_feats[
                "all_atom_positions"
            ]  # cache `atom23` positions
        
        if na_inputs_present:
            na_chain_feats = data_transforms.make_atom23_masks(na_chain_feats)
            data_transforms.atom23_list_to_atom27_list(
                na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
            )
            na_chain_feats = data_transforms.atom27_to_frames(na_chain_feats)
            na_chain_feats = data_transforms.atom27_to_torsion_angles()(na_chain_feats)

        # Merge available protein and nucleic acid features using padding where necessary
        chain_feats = du.concat_complex_torch_features(
            chain_feats,
            {}, # empty protein features
            na_chain_feats,
            feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
            add_batch_dim=False,
        )

        # cleaner version
        final_feats = {
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
            "is_na_residue_mask": processed_feats["is_na_residue_mask"],
            "aatype": chain_feats["aatype"],
        }

        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
                                        chain_feats["rigidgroups_gt_frames"]
                                    )[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        final_feats["rotmats_1"] = rotmats_1
        final_feats["trans_1"] = trans_1
        final_feats['res_mask'] = torch.tensor(processed_feats['bb_mask']).int()

        return final_feats

    def convert_dict_float64_items_to_float32(self, dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                converted_dict[key] = value  # For non-NumPy array and non-PyTorch tensor types
        return converted_dict
    
    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row["processed_path"]
        final_feats = self._process_csv_row(processed_file_path)

        max_len = int(csv_row["modeled_na_seq_len"])

    # 先准备 bp label，但先不要塞进 final_feats
        n_res = max_len
        bp_matrix = np.zeros((n_res, n_res), dtype=np.float32)
        bp_label_mask = False

        bp_root = getattr(self._data_conf, 'bp_labels_root', None)
        if bp_root:
            bp_path = os.path.join(bp_root, 'labels', f"{csv_row['pdb_name']}.bp.npz")
            if os.path.isfile(bp_path):
                try:
                    bp_npz = np.load(bp_path, allow_pickle=False)
                    bp_arr = bp_npz['bp_matrix'].astype(np.float32)
                    if bp_arr.shape == (n_res, n_res):
                        bp_matrix = bp_arr
                        bp_label_mask = True
                except Exception:
                    pass

    # 先 pad 普通结构 features
        final_feats = tree.map_structure(
        lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats
         )
        final_feats = du.pad_feats(final_feats, max_len)
        final_feats = self.convert_dict_float64_items_to_float32(final_feats)

        # Map aatype to 4-class RNA sequence target (A=0, C=1, G=2, U=3), ignore others.
        if 'aatype' in final_feats:
            raw_aatype = torch.as_tensor(final_feats['aatype']).long()
            seq_target = torch.full_like(raw_aatype, -100)
            seq_target[raw_aatype == 21] = 0  # A
            seq_target[raw_aatype == 22] = 1  # C
            seq_target[raw_aatype == 23] = 2  # G
            seq_target[raw_aatype == 25] = 3  # U
            # Mask out padding positions.
            seq_target[final_feats['res_mask'] == 0] = -100
            final_feats['aatype'] = seq_target

         # 再手动加入 bp label
        bp_matrix_padded = np.zeros((max_len, max_len), dtype=np.float32)
        bp_matrix_padded[:n_res, :n_res] = bp_matrix

        final_feats['bp_matrix'] = torch.tensor(bp_matrix_padded, dtype=torch.float32)
        final_feats['bp_label_mask'] = torch.tensor(bp_label_mask, dtype=torch.bool)

        return final_feats

    def __len__(self):
        return len(self.csv)

class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )

        # ignore the above variable if subset is given
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_subset]
        
        print (f"#### Generating sequences with the following lengths: {list(all_sample_lengths)}")

        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self._samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch
