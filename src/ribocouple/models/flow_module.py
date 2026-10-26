"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_module.py
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
import random
from datetime import datetime
import wandb
import numpy as np
import pandas as pd
import logging
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pytorch_lightning import LightningModule

from ribocouple.analysis import metrics
from ribocouple.analysis import utils as au
from ribocouple.models.flow_model import FlowModel
from ribocouple.models import utils as mu
from ribocouple.data.interpolant import Interpolant 
from ribocouple.data import utils as du
from ribocouple.data import all_atom as rna_all_atom
from ribocouple.data import so3_utils
from ribocouple.data import nucleotide_constants
from ribocouple.analysis.validation_eval import ValidationExternalEvaluator
from pytorch_lightning.loggers.wandb import WandbLogger

class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        self._validation_cfg = getattr(self._exp_cfg, 'validation', None)
        self._validation_samples_root = os.path.join(self._sample_write_dir, 'validation_samples')
        os.makedirs(self._validation_samples_root, exist_ok=True)
        self._external_eval_dir = os.path.join(self._sample_write_dir, 'validation_external_eval')
        os.makedirs(self._external_eval_dir, exist_ok=True)
        self._external_evaluator = None
        if (
            self._validation_cfg is not None and
            getattr(self._validation_cfg, 'external_eval', None) is not None and
            bool(self._validation_cfg.external_eval.enabled)
        ):
            self._external_evaluator = ValidationExternalEvaluator(self._validation_cfg.external_eval)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.validation_epoch_sample_paths = []
        self.validation_epoch_sample_scores = []
        self.validation_epoch_pair_metrics = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch):
        """
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects

        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones

        Returns:
            Dictionary of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape

        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index

        if training_cfg.num_non_frame_atoms == 0:
            bb_filtered_atom_idx = [2, 3, 6] # [C3', C4', O4']
        elif training_cfg.num_non_frame_atoms == 3:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 7, 9] # [C3', C4', O4'] + [C1', O3', P]
        elif training_cfg.num_non_frame_atoms == 7:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 4, 7, 9, 10, 11, 12] # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
        else:
            # NOTE: default is the original frame
            bb_filtered_atom_idx = [2, 3, 6] # [C3', C4', O4']
        
        n_merged_atoms = len(bb_filtered_atom_idx)

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_torsions_1 = noisy_batch['torsion_angles_sin_cos'][:, :, torsions_start_index:torsions_end_index, :].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        gt_bb_atoms = rna_all_atom.to_atom37_rna(
                            gt_trans_1, gt_rotmats_1, 
                            torch.ones_like(is_na_residue_mask),
                            torsions=gt_torsions_1
                        )
        gt_bb_atoms = gt_bb_atoms[:, :, bb_filtered_atom_idx]

        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        topology_feedback_stats = model_output.get('topology_feedback_stats', None)
        pair_recycle_stats = model_output.get('pair_recycle_stats', None)
        if topology_feedback_stats is not None:
            self._log_scalar(
                'train/topology_feedback_block',
                float(topology_feedback_stats['block']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
            self._log_scalar(
                'train/topology_pair_prob_mean',
                float(topology_feedback_stats['pair_prob_mean']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
            self._log_scalar(
                'train/topology_pair_prob_std',
                float(topology_feedback_stats['pair_prob_std']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
            self._log_scalar(
                'train/topology_pair_prob_min',
                float(topology_feedback_stats['pair_prob_min']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
            self._log_scalar(
                'train/topology_pair_prob_max',
                float(topology_feedback_stats['pair_prob_max']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
        if pair_recycle_stats is not None:
            self._log_scalar(
                'train/pair_recycle_mean',
                float(pair_recycle_stats['pair_prob_mean']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
            self._log_scalar(
                'train/pair_recycle_std',
                float(pair_recycle_stats['pair_prob_std']),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_1 = model_output['pred_torsions'].reshape(num_batch, num_res, num_torsions * 2)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = rna_all_atom.to_atom37_rna(
                            pred_trans_1, pred_rotmats_1, 
                            torch.ones_like(is_na_residue_mask),
                            torsions=pred_torsions_1
                        )
        pred_bb_atoms = pred_bb_atoms[:, :, bb_filtered_atom_idx]
        
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * n_merged_atoms
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3]) 
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3]) 
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_merged_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_merged_atoms]) 

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * 8  # 8 torsion angles
        tors_loss = training_cfg.tors_loss_scale * torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom

        pair_bce_weight = float(getattr(training_cfg, 'pair_bce_loss_weight', 0.0))
        pair_bce_smooth = float(getattr(training_cfg, 'pair_bce_label_smoothing', 0.0))
        bp_targets = noisy_batch.get('bp_matrix', None)
        bp_label_mask = noisy_batch.get('bp_label_mask', None)
        bp_targets_raw = bp_targets
        # For pair BCE, prefer pass1 logits when pair recycle is enabled,
        # since pass1 is the explicit pair prediction and pass2 is conditioned.
        pair_logits = model_output.get('pair_logits', None)
        use_pair_recycle = getattr(self.model, '_use_pair_recycle', False)
        pass1_pair_bce = getattr(self.model, '_pair_recycle_pass1_pair_bce', False)
        if use_pair_recycle and pass1_pair_bce:
            pair_logits = model_output.get('pair_logits_pass1', pair_logits)
        if pair_bce_weight > 0 and bp_targets is not None and pair_logits is not None:
            bp_targets = bp_targets.to(pair_logits.device, dtype=pair_logits.dtype)
            if pair_bce_smooth > 0:
                bp_targets = bp_targets * (1.0 - pair_bce_smooth) + 0.5 * pair_bce_smooth

            pair_mask_2d = (
                noisy_batch['res_mask'].float()[:, :, None] * noisy_batch['res_mask'].float()[:, None, :]
            )
            pair_mask_2d = pair_mask_2d * (
                1.0 - torch.eye(num_res, device=pair_logits.device, dtype=pair_logits.dtype)[None]
            )
            n_pos = (bp_targets * pair_mask_2d).sum(dim=(1, 2), keepdim=True).clamp(min=1)
            n_neg = pair_mask_2d.sum(dim=(1, 2), keepdim=True) - n_pos
            pos_weight = (n_neg / n_pos).clamp(max=30)  # [B, 1, 1]
            pair_bce = F.binary_cross_entropy_with_logits(
                pair_logits, bp_targets, pos_weight=pos_weight, reduction='none'
            )
            pair_bce_loss = torch.sum(pair_bce * pair_mask_2d, dim=(1, 2)) / (
                torch.sum(pair_mask_2d, dim=(1, 2)) + 1e-6
            )
            if bp_label_mask is not None:
                sample_mask = bp_label_mask.to(pair_logits.device).float().reshape(-1)
                pair_bce_loss = pair_bce_loss * sample_mask
            pair_bce_loss = pair_bce_loss * pair_bce_weight
        else:
            pair_bce_loss = torch.zeros(num_batch, device=pred_trans_1.device, dtype=pred_trans_1.dtype)

        # E4: sequence prediction cross-entropy loss
        seq_ce_weight = float(getattr(training_cfg, 'seq_ce_loss_weight', 0.0))
        seq_logits = model_output.get('seq_logits', None)
        aatype = noisy_batch.get('aatype', None)
        if seq_ce_weight > 0 and seq_logits is not None and aatype is not None:
            seq_ce = F.cross_entropy(
                seq_logits.transpose(1, 2),  # [B, num_classes, L]
                aatype,  # [B, L]
                ignore_index=-100,
                reduction='none',
            )  # [B, L]
            seq_ce_loss = torch.sum(seq_ce * loss_mask, dim=-1) / (
                torch.sum(loss_mask, dim=-1) + 1e-6
            )
            seq_ce_loss = seq_ce_loss * seq_ce_weight
        else:
            seq_ce_loss = torch.zeros(num_batch, device=pred_trans_1.device, dtype=pred_trans_1.dtype)

        # E4.5: conditional pair-type loss on true paired positions only.
        pair_type_ce_weight = float(getattr(training_cfg, 'pair_type_ce_loss_weight', 0.0))
        pair_type_logits = model_output.get('pair_type_logits', None)
        if (
            pair_type_ce_weight > 0 and
            pair_type_logits is not None and
            bp_targets_raw is not None and
            aatype is not None
        ):
            pair_type_labels = self._make_pair_type_labels(
                aatype.to(pair_type_logits.device),
                bp_targets_raw.to(pair_type_logits.device),
            )
            pair_type_valid = pair_type_labels >= 0
            if bp_label_mask is not None:
                pair_type_valid = pair_type_valid & (
                    bp_label_mask.to(pair_type_logits.device).bool().view(-1, 1, 1)
                )
            pair_type_labels_ce = pair_type_labels.masked_fill(~pair_type_valid, -100)
            class_weights = self._pair_type_class_weights(
                pair_type_labels_ce,
                pair_type_logits.shape[-1],
                pair_type_logits.device,
                pair_type_logits.dtype,
            )
            pair_type_ce = F.cross_entropy(
                pair_type_logits.permute(0, 3, 1, 2),
                pair_type_labels_ce,
                weight=class_weights,
                ignore_index=-100,
                reduction='none',
            )
            pair_type_den = pair_type_valid.float().sum(dim=(1, 2)).clamp(min=1.0)
            pair_type_ce_loss = (
                pair_type_ce * pair_type_valid.float()
            ).sum(dim=(1, 2)) / pair_type_den
            pair_type_ce_loss = pair_type_ce_loss * pair_type_ce_weight

            with torch.no_grad():
                pred_pair_type = torch.argmax(pair_type_logits, dim=-1)
                # All DDP ranks must execute the same sync_dist logging call.
                # A local batch may contain no labelled positive pair, so never
                # guard this log with `if pair_type_valid.any()`.
                pair_type_correct = (
                    (pred_pair_type == pair_type_labels) & pair_type_valid
                ).float().sum()
                pair_type_count = pair_type_valid.float().sum()
                pair_type_acc = (
                    pair_type_correct / pair_type_count.clamp(min=1.0)
                )
                self._log_scalar(
                    'train/pair_type_acc',
                    pair_type_acc,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=True,
                    rank_zero_only=False,
                )
        else:
            pair_type_ce_loss = torch.zeros(num_batch, device=pred_trans_1.device, dtype=pred_trans_1.dtype)

        # E4.6: differentiable pair-type / sequence consistency loss.
        # Pair-type probabilities are detached: this term trains the sequence
        # head to realize the predicted chemistry, instead of letting the pair
        # head change its prediction to avoid the penalty.
        pair_type_consistency_weight = float(
            getattr(training_cfg, 'pair_type_consistency_loss_weight', 0.0)
        )
        pair_type_consistency_loss = torch.zeros(
            num_batch, device=pred_trans_1.device, dtype=pred_trans_1.dtype
        )
        if (
            pair_type_consistency_weight > 0
            and seq_logits is not None
            and pair_type_logits is not None
            and bp_targets_raw is not None
            and aatype is not None
        ):
            pair_type_consistency_loss = self._compute_pair_type_consistency_loss(
                seq_logits=seq_logits,
                pair_logits=pair_logits,
                pair_type_logits=pair_type_logits,
                bp_targets=bp_targets_raw,
                aatype=aatype,
                res_mask=loss_mask,
                pair_label_mask=bp_label_mask,
                pair_threshold=float(getattr(
                    training_cfg, 'pair_type_consistency_pair_threshold', 0.5
                )),
                type_conf_threshold=float(getattr(
                    training_cfg, 'pair_type_consistency_conf_threshold', 0.5
                )),
            )
            pair_type_consistency_loss = (
                pair_type_consistency_loss * pair_type_consistency_weight
            )

        assert bb_atom_loss.shape[0] == dist_mat_loss.shape[0] == tors_loss.shape[0] == pair_bce_loss.shape[0] == seq_ce_loss.shape[0] == pair_type_ce_loss.shape[0] == pair_type_consistency_loss.shape[0], f"Loss tensors shape mismatch: {bb_atom_loss.shape} vs {dist_mat_loss.shape} vs {tors_loss.shape} vs {pair_bce_loss.shape} vs {seq_ce_loss.shape} vs {pair_type_ce_loss.shape} vs {pair_type_consistency_loss.shape}"

        # E4: sequence regularization losses (GC, MFE surrogate, pair WC complementarity)
        seq_reg_losses = self._compute_seq_regularization_losses(
            seq_logits, pair_logits, loss_mask
        )

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss + tors_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight

        if torch.isnan(auxiliary_loss).any():
            print ("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)

        if torch.isnan(se3_vf_loss).any():
            # raise ValueError('NaN loss encountered')
            print ("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)

        # Sequence recovery metric (for logging)
        if seq_logits is not None and aatype is not None:
            pred_seq = torch.argmax(seq_logits, dim=-1)  # [B, L]
            valid_mask = (aatype >= 0) & (aatype < self.model._seq_head_num_classes) & loss_mask.bool()
            if valid_mask.any():
                seq_recovery = (pred_seq == aatype).float()[valid_mask].mean()
                self._log_scalar(
                    'train/seq_recovery',
                    seq_recovery,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=True,
                    rank_zero_only=False,
                )

        losses = {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
            "pair_bce_loss": pair_bce_loss,
            "seq_ce_loss": seq_ce_loss,
            "pair_type_ce_loss": pair_type_ce_loss,
            "pair_type_consistency_loss": pair_type_consistency_loss,
        }
        losses.update(seq_reg_losses)
        return losses

    @staticmethod
    def _compute_generated_pair_type_metrics(
        pair_logits,
        pair_type_logits,
        seq_logits,
        res_mask,
        pair_threshold=0.5,
        type_conf_threshold=0.0,
    ):
        """Metrics for whether sampled sequences realize predicted pair types."""
        zero = {
            "pair_type_realization": 0.0,
            "au_realization": 0.0,
            "gc_realization": 0.0,
            "gu_realization": 0.0,
            "other_realization": 0.0,
            "pair_type_coverage": 0.0,
        }
        if pair_logits is None or pair_type_logits is None or seq_logits is None:
            return zero

        pair_prob = torch.sigmoid(pair_logits)
        type_prob = torch.softmax(pair_type_logits, dim=-1)
        pred_type = type_prob.argmax(dim=-1)
        type_conf = type_prob.max(dim=-1).values
        pred_seq = seq_logits.argmax(dim=-1)
        # A=0, C=1, G=2, U=3 in the sequence head.
        ai = pred_seq.unsqueeze(2)
        aj = pred_seq.unsqueeze(1)
        seq_type = torch.full_like(pred_type, -1)
        seq_type[((ai == 0) & (aj == 3)) | ((ai == 3) & (aj == 0))] = 0  # AU
        seq_type[((ai == 1) & (aj == 2)) | ((ai == 2) & (aj == 1))] = 1  # GC
        seq_type[((ai == 2) & (aj == 3)) | ((ai == 3) & (aj == 2))] = 2  # GU
        seq_type[seq_type < 0] = 3  # OTHER

        B, L, _ = pair_prob.shape
        mask = res_mask.bool().unsqueeze(1) & res_mask.bool().unsqueeze(2)
        mask &= ~torch.eye(L, device=pair_prob.device, dtype=torch.bool).unsqueeze(0)
        mask &= pair_prob >= pair_threshold
        if type_conf_threshold > 0:
            mask &= type_conf >= type_conf_threshold

        vals = {}
        denom = mask.float().sum().clamp(min=1.0)
        correct = (pred_type == seq_type) & mask
        vals["pair_type_realization"] = (correct.float().sum() / denom).detach()
        vals["pair_type_coverage"] = mask.float().mean().detach()
        for idx, name in ((0, "au"), (1, "gc"), (2, "gu"), (3, "other")):
            class_mask = mask & (pred_type == idx)
            class_denom = class_mask.float().sum().clamp(min=1.0)
            class_correct = class_mask & (seq_type == idx)
            vals[f"{name}_realization"] = (
                class_correct.float().sum() / class_denom
            ).detach()
        return vals

    def validation_step(self, batch, batch_idx):
        """
        Generates samples and computes (average) local structural metrics 
        such as C4'-C4' distances, steric clashes, gyration radius.
        """

        res_mask = batch['res_mask']
        is_na_residue_mask = batch['is_na_residue_mask'].bool()
        self.interpolant.set_device(res_mask.device)
        num_batch = res_mask.shape[0]
        num_res = is_na_residue_mask.sum(dim=-1).max().item()
        rank_tag = f'rank_{int(getattr(self, "global_rank", 0)):02d}'
        
        sample_output = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            return_model_out=True,
        )
        samples = sample_output[0][-1].numpy()
        model_out_sample = sample_output[3]
        seq_logits_sample = model_out_sample.get('seq_logits', None) if model_out_sample is not None else None
        if model_out_sample is not None:
            generated_pair_type_metrics = self._compute_generated_pair_type_metrics(
                pair_logits=model_out_sample.get('pair_logits', None),
                pair_type_logits=model_out_sample.get('pair_type_logits', None),
                seq_logits=seq_logits_sample,
                res_mask=res_mask,
                pair_threshold=float(getattr(self._exp_cfg.training, 'pair_type_consistency_pair_threshold', 0.5)),
                type_conf_threshold=0.0,
            )
            for metric_name, metric_value in generated_pair_type_metrics.items():
                self.log(
                    f'valid/{metric_name}',
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=True,
                )
        noisy_batch = self.interpolant.corrupt_batch(batch)
        with torch.no_grad():
            model_output = self.model(noisy_batch)
        pair_metrics = self._compute_pair_metrics(
            model_output.get('pair_logits', None),
            noisy_batch.get('bp_matrix', None),
            noisy_batch.get('bp_label_mask', None),
            noisy_batch.get('res_mask', None),
        )

        # E4: validation sequence recovery from corrupted forward pass
        seq_logits = model_output.get('seq_logits', None)
        aatype = noisy_batch.get('aatype', None)
        if seq_logits is not None and aatype is not None:
            pred_seq = torch.argmax(seq_logits, dim=-1)
            valid_mask = (
                (aatype >= 0) &
                (aatype < self.model._seq_head_num_classes) &
                res_mask.bool()
            )
            if valid_mask.any():
                val_seq_recovery = (pred_seq == aatype).float()[valid_mask].mean()
                self.log(
                    'valid/seq_recovery',
                    val_seq_recovery,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=True,
                )

        # E4: validation sequence properties (GC & MFE surrogate + actual MFE)
        if seq_logits is not None:
            B, L, _ = seq_logits.shape
            device = seq_logits.device
            seq_soft = F.softmax(seq_logits, dim=-1)
            gc_prob = seq_soft[..., 1] + seq_soft[..., 2]
            valid_len = res_mask.float().sum(dim=-1).clamp(min=1)
            mean_gc = (gc_prob * res_mask.float()).sum(dim=-1) / valid_len
            self.log(
                'valid/mean_gc',
                mean_gc.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=True,
            )

            pair_logits = model_output.get('pair_logits', None)
            if pair_logits is not None:
                pair_mask_2d = (
                    res_mask.float()[:, :, None] * res_mask.float()[:, None, :]
                )
                pair_mask_2d = pair_mask_2d * (
                    1.0 - torch.eye(L, device=device, dtype=pair_mask_2d.dtype)[None]
                )
                pair_probs = torch.sigmoid(pair_logits)
                energy_pairs = [
                    (1, 2, -3.0), (2, 1, -3.0),
                    (0, 3, -2.0), (3, 0, -2.0),
                    (2, 3, -1.0), (3, 2, -1.0),
                ]
                expected_energy = torch.zeros((B, L, L), device=device)
                for a, b, e in energy_pairs:
                    p_a_i = seq_soft[:, :, a].unsqueeze(2)
                    p_b_j = seq_soft[:, :, b].unsqueeze(1)
                    expected_energy += e * p_a_i * p_b_j
                pair_energy = (pair_probs * expected_energy * pair_mask_2d).sum(dim=(1, 2))
                energy_per_nt = pair_energy / valid_len
                self.log(
                    'valid/mfe_surrogate_energy',
                    energy_per_nt.mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=True,
                )

            # Actual MFE via ViennaRNA (for monitoring only)
            try:
                import RNA
                pred_seq = torch.argmax(seq_logits, dim=-1).cpu().numpy()
                mfe_list = []
                for i in range(B):
                    length = int(valid_len[i].item())
                    seq_chars = ['ACGU'[idx] for idx in pred_seq[i][:length]]
                    seq_str = ''.join(seq_chars)
                    if len(seq_str) > 0:
                        _, mfe = RNA.fold(seq_str)
                        mfe_list.append(mfe)
                if mfe_list:
                    self.log(
                        'valid/mfe_actual',
                        np.mean(mfe_list),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=num_batch,
                        sync_dist=True,
                    )
            except Exception:
                pass

        batch_metrics = []
        should_save_pdbs = (
            True
            if self._validation_cfg is None
            else bool(self._validation_cfg.save_pdbs or self._external_evaluator is not None)
        )
        epoch_tag = f'epoch_{self.current_epoch:04d}'
        step_tag = f'step_{self.global_step:08d}'
        time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        val_subdir = os.path.join(
            self._validation_samples_root,
            f'{epoch_tag}_{step_tag}_{time_tag}_{rank_tag}',
        )
        if should_save_pdbs:
            os.makedirs(val_subdir, exist_ok=True)

        for i in range(num_batch):
            final_pos = samples[i]
            saved_rna_path = None
            if should_save_pdbs:
                saved_rna_path = au.write_complex_to_pdbs( # Save RNA atoms to PDB
                    final_pos,
                    os.path.join(
                        val_subdir,
                        f'sample_{i}_idx_{batch_idx}_len_{num_res}_epoch_{self.current_epoch}_step_{self.global_step}',
                    ),
                    is_na_residue_mask=is_na_residue_mask.detach().cpu().numpy()[i],
                )
                self.validation_epoch_sample_paths.append(saved_rna_path)
                # E4: save predicted sequence as FASTA for external RhoFold evaluation
                if seq_logits_sample is not None:
                    pred_seq_indices = torch.argmax(seq_logits_sample[i], dim=-1).cpu().numpy()
                    pred_seq = ''.join(
                        'ACGU'[idx] for idx in pred_seq_indices
                        if 0 <= idx < self.model._seq_head_num_classes
                    )
                    fasta_path = saved_rna_path.replace('.pdb', '.fasta')
                    self._write_single_seq_fasta(
                        pred_seq,
                        os.path.basename(saved_rna_path).replace('.pdb', ''),
                        fasta_path,
                    )

                # Pair-score for optional reranking
                if (
                    getattr(self._validation_cfg, 'use_pair_rerank', False) and
                    model_out_sample is not None and
                    'pair_logits' in model_out_sample
                ):
                    pair_logits_i = model_out_sample['pair_logits'][i]
                    res_mask_i = res_mask[i].bool()
                    pair_score = self._compute_topk_pair_sum(pair_logits_i, res_mask_i)
                    self.validation_epoch_sample_scores.append(
                        (saved_rna_path, pair_score, num_res)
                    )

                if isinstance(self.logger, WandbLogger):
                    self.validation_epoch_samples.append(
                        [saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)]
                    )

            c4_idx = nucleotide_constants.atom_order["C4\'"]
            rna_c4_c4_matrics = metrics.calc_rna_c4_c4_metrics(final_pos[:, c4_idx])
            batch_metrics.append(rna_c4_c4_matrics)

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        if pair_metrics:
            self.validation_epoch_pair_metrics.append(pd.DataFrame(pair_metrics))
        
    def on_validation_epoch_end(self):
        if (
            self.trainer is not None and
            self.trainer.is_global_zero and
            isinstance(self.logger, WandbLogger) and
            len(self.validation_epoch_samples) > 0
        ):
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "RNA"],
                data=self.validation_epoch_samples)
        self.validation_epoch_samples.clear()

        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        gathered_sample_paths = self._gather_sample_paths(self.validation_epoch_sample_paths)

        # Optional pair-score reranking before external eval.
        use_pair_rerank = bool(getattr(self._validation_cfg, 'use_pair_rerank', False))
        if use_pair_rerank:
            gathered_sample_scores = self._gather_sample_scores(self.validation_epoch_sample_scores)
            rerank_topk = int(getattr(self._validation_cfg, 'rerank_topk', 10))
            n_before = len(gathered_sample_paths)
            gathered_sample_paths = self._rerank_sample_paths(
                gathered_sample_paths, gathered_sample_scores, rerank_topk
            )
            n_after = len(gathered_sample_paths)
            if self.trainer is None or self.trainer.is_global_zero:
                print(f"[E4 Rerank] selected {n_after}/{n_before} samples (top-{rerank_topk} per length)")
                self._log_scalar(
                    'valid/rerank_selected_ratio',
                    n_after / max(n_before, 1),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=1,
                    sync_dist=False,
                    rank_zero_only=True,
                )

        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
                sync_dist=True,
                rank_zero_only=False,
            )

        if len(self.validation_epoch_pair_metrics) > 0:
            pair_epoch_metrics = pd.concat(self.validation_epoch_pair_metrics)
            for metric_name, metric_val in pair_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(pair_epoch_metrics),
                    sync_dist=True,
                    rank_zero_only=False,
                )

        ext_summary = None
        if (
            self._external_evaluator is not None and
            self.trainer is not None and
            self.trainer.is_global_zero and
            len(gathered_sample_paths) > 0 and
            self._should_run_external_eval()
        ):
            eval_dir = os.path.join(
                self._external_eval_dir,
                f'epoch_{self.current_epoch:04d}_step_{self.global_step:08d}',
            )
            ext_summary = self._external_evaluator.evaluate(
                gathered_sample_paths,
                eval_dir,
            )
        elif self._external_evaluator is not None and self.trainer is not None and self.trainer.is_global_zero:
            ext_summary = self._external_evaluator._default_summary()
            ext_summary['ext_n_backbones'] = float(len(gathered_sample_paths))
            ext_summary['ext_n_success'] = 0.0

        ext_summary = self._broadcast_object(ext_summary)

        if ext_summary is not None:
            ext_summary['ext_eval_ran'] = float(
                1.0 if self._should_run_external_eval() and len(gathered_sample_paths) > 0 else 0.0
            )
            for metric_name, metric_val in ext_summary.items():
                if isinstance(metric_val, str):
                    # Skip string metadata fields (e.g., ext_sequence_source).
                    continue
                self._log_scalar(
                    f'valid/{metric_name}',
                    float(metric_val),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=max(len(gathered_sample_paths), 1),
                    sync_dist=False,
                    rank_zero_only=False,
                )

        self.validation_epoch_metrics.clear()
        self.validation_epoch_sample_paths.clear()
        self.validation_epoch_sample_scores.clear()
        self.validation_epoch_pair_metrics.clear()

    def _compute_pair_metrics(self, pair_logits, bp_targets, bp_label_mask, res_mask):
        if pair_logits is None or bp_targets is None or res_mask is None:
            return []

        pair_probs = torch.sigmoid(pair_logits)
        pair_preds = pair_probs >= 0.5
        pair_targets = bp_targets > 0.5

        res_mask = res_mask.bool()
        num_batch, num_res = res_mask.shape
        eye = torch.eye(num_res, device=res_mask.device, dtype=torch.bool)[None]
        pair_mask = res_mask[:, :, None] & res_mask[:, None, :] & (~eye)
        upper_mask = torch.triu(torch.ones(num_res, num_res, device=res_mask.device, dtype=torch.bool), diagonal=1)[None]
        pair_mask = pair_mask & upper_mask

        if bp_label_mask is not None:
            sample_mask = bp_label_mask.bool().reshape(-1)
        else:
            sample_mask = torch.ones(num_batch, device=res_mask.device, dtype=torch.bool)

        rows = []
        for i in range(num_batch):
            if not sample_mask[i]:
                continue
            mask = pair_mask[i]
            target = pair_targets[i][mask]
            pred = pair_preds[i][mask]
            prob = pair_probs[i][mask]
            if target.numel() == 0:
                continue

            tp = torch.sum(pred & target).item()
            fp = torch.sum(pred & (~target)).item()
            fn = torch.sum((~pred) & target).item()
            tn = torch.sum((~pred) & (~target)).item()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            pos_rate = torch.mean(target.float()).item()
            pred_rate = torch.mean(pred.float()).item()
            prob_mean = torch.mean(prob.float()).item()

            rows.append(
                {
                    'pair_precision': precision,
                    'pair_recall': recall,
                    'pair_f1': f1,
                    'pair_accuracy': acc,
                    'pair_positive_rate': pos_rate,
                    'pair_pred_positive_rate': pred_rate,
                    'pair_prob_mean': prob_mean,
                }
            )
        return rows

    def _gather_sample_paths(self, local_paths):
        local_paths = list(local_paths)
        if not dist.is_available() or not dist.is_initialized():
            return local_paths

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_paths)
        merged = []
        for paths in gathered:
            if not paths:
                continue
            merged.extend(paths)
        return merged

    def _gather_sample_scores(self, local_scores):
        local_scores = list(local_scores)
        if not dist.is_available() or not dist.is_initialized():
            return local_scores

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_scores)
        merged = []
        for scores in gathered:
            if not scores:
                continue
            merged.extend(scores)
        return merged

    def _rerank_sample_paths(self, sample_paths, sample_scores, topk):
        """Select top-K samples per length by pair score."""
        from collections import defaultdict

        if not sample_scores:
            return sample_paths

        # sample_scores: list of (path, score, length)
        by_length = defaultdict(list)
        for path, score, length in sample_scores:
            by_length[length].append((path, score))

        selected = set()
        for length, items in by_length.items():
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            k = min(topk, len(items_sorted))
            for path, _ in items_sorted[:k]:
                selected.add(path)

        # Preserve original order for selected paths.
        return [p for p in sample_paths if p in selected]

    def _broadcast_object(self, obj):
        if not dist.is_available() or not dist.is_initialized():
            return obj

        payload = [obj if self.trainer is not None and self.trainer.is_global_zero else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _should_run_external_eval(self):
        if self._validation_cfg is None:
            return False
        ext_cfg = getattr(self._validation_cfg, 'external_eval', None)
        if ext_cfg is None or not bool(ext_cfg.enabled):
            return False
        every_n = int(getattr(ext_cfg, 'every_n_epochs', 1))
        if every_n <= 0:
            every_n = 1
        return (self.current_epoch + 1) % every_n == 0

    @staticmethod
    def _write_single_seq_fasta(seq: str, name: str, path: str) -> str:
        SeqIO.write([SeqRecord(Seq(seq), id=name, description="")], path, "fasta")
        return path

    @staticmethod
    def _compute_topk_pair_sum(pair_logits, res_mask, topk=10):
        """Compute sum of top-K pair probabilities as a reranking score."""
        pair_probs = torch.sigmoid(pair_logits)
        L = pair_logits.shape[-1]
        device = pair_logits.device
        eye = torch.eye(L, device=device).bool()
        upper = torch.triu(torch.ones(L, L, device=device).bool(), diagonal=1)
        mask = res_mask[:, None] & res_mask[None, :] & upper & ~eye
        probs_masked = pair_probs.masked_fill(~mask, 0.0)
        k = min(topk, int(mask.sum().item()))
        if k > 0:
            return probs_masked.view(-1).topk(k).values.sum().item()
        return 0.0

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def _compute_seq_regularization_losses(self, seq_logits, pair_logits, loss_mask):
        """
        E4 sequence head regularization:
        - GC content penalty
        - MFE surrogate via predicted pair energies
        - WC complementarity for high-probability predicted pairs
        Returns a dict of per-sample loss tensors [B].
        """
        losses = {}
        seq_head_cfg = getattr(self._model_cfg, 'sequence_head', None)
        if seq_head_cfg is None:
            return losses
        reg_cfg = getattr(seq_head_cfg, 'regularization', None)
        if reg_cfg is None:
            return losses

        B, L, _ = seq_logits.shape
        device = seq_logits.device
        # Nucleotide order: A=0, C=1, G=2, U=3
        seq_soft = F.softmax(seq_logits, dim=-1)  # [B, L, 4]
        gc_prob = seq_soft[..., 1] + seq_soft[..., 2]  # [B, L]

        gc_cfg = reg_cfg.get('gc', None)
        if gc_cfg and gc_cfg.get('enabled', False):
            target_gc = float(gc_cfg.get('target_gc', 0.5))
            lambda_gc = float(gc_cfg.get('lambda_gc', 0.0))
            valid_len = loss_mask.sum(dim=-1).clamp(min=1)
            mean_gc = (gc_prob * loss_mask).sum(dim=-1) / valid_len
            losses['seq_gc_loss'] = lambda_gc * torch.abs(mean_gc - target_gc)
            self._log_scalar(
                'train/mean_gc',
                mean_gc.mean(),
                prog_bar=False,
                batch_size=B,
                rank_zero_only=True,
            )

        if pair_logits is None:
            return losses

        pair_mask_2d = (
            loss_mask.float()[:, :, None] * loss_mask.float()[:, None, :]
        )
        pair_mask_2d = pair_mask_2d * (
            1.0 - torch.eye(L, device=device, dtype=pair_mask_2d.dtype)[None]
        )
        pair_probs = torch.sigmoid(pair_logits)

        pc_cfg = reg_cfg.get('pair_complementarity', None)
        if pc_cfg and pc_cfg.get('enabled', False):
            lambda_pair = float(pc_cfg.get('lambda_pair', 0.0))
            prob_threshold = float(pc_cfg.get('prob_threshold', 0.8))
            wc_pairs = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
            p_wc = torch.zeros((B, L, L), device=device)
            for a, b in wc_pairs:
                p_a_i = seq_soft[:, :, a].unsqueeze(2)
                p_b_j = seq_soft[:, :, b].unsqueeze(1)
                p_wc += p_a_i * p_b_j
            high_prob_mask = (pair_probs >= prob_threshold).float() * pair_mask_2d
            denom = high_prob_mask.sum(dim=(1, 2)).clamp(min=1)
            losses['seq_pair_comp_loss'] = (
                lambda_pair * ((1.0 - p_wc) * high_prob_mask).sum(dim=(1, 2)) / denom
            )

        mfe_cfg = reg_cfg.get('mfe', None)
        if mfe_cfg and mfe_cfg.get('enabled', False):
            lambda_mfe = float(mfe_cfg.get('lambda_mfe', 0.0))
            threshold_per_nt = float(mfe_cfg.get('threshold_per_nt', -0.3))
            # Nussinov-like pair energies: GC=-3, AU=-2, GU=-1
            energy_pairs = [
                (1, 2, -3.0), (2, 1, -3.0),
                (0, 3, -2.0), (3, 0, -2.0),
                (2, 3, -1.0), (3, 2, -1.0),
            ]
            expected_energy = torch.zeros((B, L, L), device=device)
            for a, b, e in energy_pairs:
                p_a_i = seq_soft[:, :, a].unsqueeze(2)
                p_b_j = seq_soft[:, :, b].unsqueeze(1)
                expected_energy += e * p_a_i * p_b_j
            pair_energy = (pair_probs * expected_energy * pair_mask_2d).sum(dim=(1, 2))
            seq_len = loss_mask.sum(dim=-1).clamp(min=1).float()
            energy_per_nt = pair_energy / seq_len
            # threshold_per_nt is a negative number (e.g. -0.5 kcal/mol/nt).
            # Penalize if the surrogate energy per nt is more negative than threshold.
            losses['seq_mfe_loss'] = (
                lambda_mfe * torch.clamp(threshold_per_nt - energy_per_nt, min=0.0)
            )
            self._log_scalar(
                'train/mfe_surrogate_energy',
                energy_per_nt.mean(),
                prog_bar=False,
                batch_size=B,
                rank_zero_only=True,
            )

        return losses

    def training_step(self, batch, stage):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        
        for k,v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar("train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        
        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)
        
        train_loss = (
            total_losses[self._exp_cfg.training.loss] +
            total_losses['auxiliary_loss'] +
            total_losses.get('pair_bce_loss', 0.0) +
            total_losses.get('seq_ce_loss', 0.0) +
            total_losses.get('pair_type_ce_loss', 0.0) +
            total_losses.get('pair_type_consistency_loss', 0.0) +
            total_losses.get('seq_gc_loss', 0.0) +
            total_losses.get('seq_mfe_loss', 0.0) +
            total_losses.get('seq_pair_comp_loss', 0.0)
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def configure_optimizers(self):
        # E4: optionally freeze trunk for post-hoc sequence head training.
        seq_head_conf = getattr(self._model_cfg, 'sequence_head', None)
        freeze_trunk = (
            seq_head_conf is not None and
            bool(getattr(seq_head_conf, 'enabled', False)) and
            bool(getattr(seq_head_conf, 'freeze_trunk', False))
        )
        if freeze_trunk:
            for name, param in self.model.named_parameters():
                if 'seq_pred_net' not in name and 'pair_type_pred_net' not in name:
                    param.requires_grad = False
            params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"[E4] Post-hoc mode: {sum(p.numel() for p in params)} head params trainable, trunk frozen.")
        else:
            params = self.model.parameters()
        return torch.optim.AdamW(
            params=params,
            **self._exp_cfg.optimizer
        )

    @staticmethod
    def _make_pair_type_labels(aatype, bp_matrix):
        """Return AU/GC/GU/OTHER labels for true paired positions, -100 elsewhere."""
        labels = torch.full(
            bp_matrix.shape,
            -100,
            dtype=torch.long,
            device=bp_matrix.device,
        )
        paired = bp_matrix > 0.5
        ai = aatype[:, :, None].expand_as(bp_matrix)
        aj = aatype[:, None, :].expand_as(bp_matrix)
        valid_base = (ai >= 0) & (ai < 4) & (aj >= 0) & (aj < 4)
        valid_pair = paired & valid_base
        labels[valid_pair] = 3  # OTHER

        au = ((ai == 0) & (aj == 3)) | ((ai == 3) & (aj == 0))
        gc = ((ai == 2) & (aj == 1)) | ((ai == 1) & (aj == 2))
        gu = ((ai == 2) & (aj == 3)) | ((ai == 3) & (aj == 2))
        labels[valid_pair & au] = 0
        labels[valid_pair & gc] = 1
        labels[valid_pair & gu] = 2
        return labels

    @staticmethod
    def _pair_type_class_weights(labels, num_classes, device, dtype):
        valid = labels >= 0
        if not valid.any():
            return None
        counts = torch.bincount(
            labels[valid].reshape(-1),
            minlength=num_classes,
        ).to(device=device, dtype=dtype)
        weights = counts.sum() / counts.clamp(min=1.0)
        weights = weights / weights.mean().clamp(min=1e-6)
        return weights

    @staticmethod
    def _compute_pair_type_consistency_loss(
        seq_logits,
        pair_logits,
        pair_type_logits,
        bp_targets,
        aatype,
        res_mask,
        pair_label_mask=None,
        pair_threshold=0.5,
        type_conf_threshold=0.5,
    ):
        """Make sequence probabilities realize predicted AU/GC/GU chemistry.

        Class convention in this project is A=0, G=1, C=2, U=3 and pair
        classes are AU=0, GC=1, GU=2, OTHER=3.
        Returns one loss value per sample, shape [B].
        """
        seq_prob = torch.softmax(seq_logits, dim=-1)
        # Do not backpropagate through pair predictions from this auxiliary
        # objective. The sequence head must follow the topology signal.
        type_prob = torch.softmax(pair_type_logits.detach(), dim=-1)
        pair_prob = torch.sigmoid(pair_logits.detach())

        # Pair chemistry probabilities under independent residue marginals.
        p_i = seq_prob[:, :, None, :]
        p_j = seq_prob[:, None, :, :]
        p_au = p_i[..., 0] * p_j[..., 3] + p_i[..., 3] * p_j[..., 0]
        p_gc = p_i[..., 1] * p_j[..., 2] + p_i[..., 2] * p_j[..., 1]
        p_gu = p_i[..., 1] * p_j[..., 3] + p_i[..., 3] * p_j[..., 1]
        realized_prob = torch.stack([p_au, p_gc, p_gu], dim=-1)

        # Predicted type distribution, excluding OTHER because OTHER has no
        # single sequence-compatibility target.
        type_prob_3 = type_prob[..., :3]
        type_conf, _ = type_prob.max(dim=-1)
        target_prob = (type_prob_3 * realized_prob).sum(dim=-1)

        valid = (bp_targets.to(seq_logits.device) > 0.5)
        valid = valid & (pair_prob >= pair_threshold)
        valid = valid & (type_conf >= type_conf_threshold)
        valid = valid & (type_prob_3.sum(dim=-1) > 1e-6)
        valid = valid & (res_mask[:, :, None].bool() & res_mask[:, None, :].bool())
        valid = valid & ~torch.eye(
            valid.shape[1], device=valid.device, dtype=torch.bool
        )[None]
        if pair_label_mask is not None:
            valid = valid & pair_label_mask.to(valid.device).bool().view(-1, 1, 1)

        loss = -torch.log(target_prob.clamp(min=1e-6))
        loss = loss * valid.float()
        denom = valid.float().sum(dim=(1, 2)).clamp(min=1.0)
        return loss.sum(dim=(1, 2)) / denom

    def predict_step(self, batch):
        """
        Params:
            batch (dict) : dictionary of number of RNA samples and respective number of RNA nucleotides to model

        Remarks:
            Used for unconditional sampling from the Interpolant. 
            Saves the generated all-atom backbones as PDB files at the specified saving directory in the inference config file.
        """

        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        diffuse_mask = torch.ones(1, sample_length).long().unsqueeze(0)
        sample_id = batch['sample_id'].item()
        sample_dir = os.path.join(self._output_dir, f'length_{sample_length}')
        os.makedirs(sample_dir, exist_ok=True)

        atom37_traj, _, _ = interpolant.sample(1, sample_length, self.model)
        bb_traj = du.to_numpy(torch.concat(atom37_traj, dim=0)) # convert from torch to numpy for easier processing

        sample = bb_traj[-1] # store last state (at the final ODESolve timestep) as completed sample
        
        # store RNA backbone sample as PDB file at `saved_rna_path` specified above
        au.write_complex_to_pdbs(
                sample,
                os.path.join(sample_dir, f'sample_{sample_id}'),
                is_na_residue_mask=diffuse_mask.detach().cpu(),
            )
        
        au.write_complex_to_pdbs(
                bb_traj,
                os.path.join(sample_dir, f'sample_{sample_id}'),
                is_na_residue_mask=diffuse_mask.detach().cpu(),
            )
