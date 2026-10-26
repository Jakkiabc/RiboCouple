"""Checkpoint loading helpers for training and inference models."""

import torch


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

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

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

    from ribocouple.models.flow_module import FlowModule

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

