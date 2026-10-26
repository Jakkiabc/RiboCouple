"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/train_se3_flows.py
"""

import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from ribocouple.data.pdb_na_datamodule_base import PDBNABaseDataModule
from ribocouple.models.flow_module import FlowModule
from ribocouple.models.callback import NanGradientCallback
import ribocouple.utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        seq_head_cfg = getattr(cfg.model, 'sequence_head', None)
        if (
            seq_head_cfg is not None
            and bool(getattr(seq_head_cfg, 'enabled', False))
            and bool(getattr(seq_head_cfg, 'freeze_trunk', False))
            and getattr(self._exp_cfg, 'warm_start', None) is None
        ):
            raise ValueError(
                'sequence_head.freeze_trunk=True requires experiment.warm_start '
                'to point to a pretrained trunk checkpoint. Set freeze_trunk=False '
                'to train the trunk from scratch.'
            )
        if (
            getattr(self._exp_cfg, 'validation', None) is not None and
            getattr(self._exp_cfg.validation, 'external_eval', None) is not None and
            not self._exp_cfg.validation.external_eval.repo_root
        ):
            self._exp_cfg.validation.external_eval.repo_root = os.getcwd()
        self._model = FlowModule(self._cfg)
        self._load_warm_start_weights()
        self._datamodule = PDBNABaseDataModule(data_cfg=self._data_cfg)

    def _load_warm_start_weights(self):
        """Load warm-start weights with strict=False when E4 sequence head is enabled.

        The E3 checkpoint does not contain the new seq_pred_net, so strict loading
        would fail. We load only model weights and start training with a fresh
        optimizer.
        """
        warm_start_path = getattr(self._exp_cfg, 'warm_start', None)
        seq_head_cfg = getattr(self._cfg.model, 'sequence_head', None)
        use_seq_head = seq_head_cfg is not None and bool(getattr(seq_head_cfg, 'enabled', False))
        if warm_start_path is None or not use_seq_head:
            return

        log.info(f"[E4] Loading warm-start weights from {warm_start_path} with strict=False")
        ckpt = torch.load(warm_start_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        allowed_missing = ('model.seq_pred_net.', 'model.pair_type_pred_net.')
        unexpected_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if unexpected_missing or unexpected:
            raise RuntimeError(
                'Warm-start checkpoint does not match the model: '
                f'unexpected missing keys={unexpected_missing[:20]}, '
                f'unexpected checkpoint keys={unexpected[:20]}'
            )
        if missing:
            log.info(f"[E4] Missing keys (expected for new sequence head): {missing[:20]}")

        # Do not resume trainer state; we want a fresh optimizer for the new head.
        self._exp_cfg.warm_start = None

    def train(self):
        callbacks = []
        
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(**self._exp_cfg.wandb,)
            
            # Checkpoint directory
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            # callbacks.append(Timer(duration=self._exp_cfg.trainer.max_time))
            callbacks.append(NanGradientCallback())
            
            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        devices = self._exp_cfg.num_devices
        log.info(f"Using devices: {devices}")
        
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )

        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
