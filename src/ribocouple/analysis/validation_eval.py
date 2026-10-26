import csv
import gc
import json
import os
import subprocess
from contextlib import contextmanager
from typing import Iterable

import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _seqrecord_to_str(seq_rec):
    if isinstance(seq_rec, SeqRecord):
        return str(seq_rec.seq).upper().replace("T", "U")
    seq = str(seq_rec).upper().replace("T", "U")
    return "".join(c if c in "ACGUN" else "N" for c in seq)


def _write_single_seq_fasta(seq: str, name: str, path: str) -> str:
    SeqIO.write([SeqRecord(Seq(seq), id=name, description="")], path, "fasta")
    return path


def _run_tmscore(tmscore_bin: str, ref_pdb: str, pred_pdb: str):
    result = subprocess.run(
        [tmscore_bin, ref_pdb, pred_pdb],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        stdout = (result.stdout or "").strip().replace("\n", " ")[:300]
        stderr = (result.stderr or "").strip().replace("\n", " ")[:300]
        raise RuntimeError(
            f"tmscore_failed_rc={result.returncode} stdout={stdout} stderr={stderr}"
        )

    tm = rmsd = gdt_ts = gdt_ha = None
    for line in result.stdout.splitlines():
        text = line.strip()
        if text.startswith("TM-score") and "=" in text and tm is None:
            tm = float(text.split("=")[1].split()[0])
        elif "RMSD of" in text and "=" in text:
            rmsd = float(text.split("=")[1].split()[0])
        elif text.startswith("GDT-TS-score") and "=" in text:
            gdt_ts = float(text.split("=")[1].split()[0])
        elif text.startswith("GDT-HA-score") and "=" in text:
            gdt_ha = float(text.split("=")[1].split()[0])

    if tm is None:
        stdout = (result.stdout or "").strip().replace("\n", " ")[:300]
        raise ValueError(f"tmscore_parse_failed stdout={stdout}")

    return {
        "tm_score": tm,
        "rmsd": rmsd,
        "gdt_ts": gdt_ts,
        "gdt_ha": gdt_ha,
    }


def _resolve_rhofold_device(cfg_value):
    """Resolve RhoFold device, allowing 'current' to follow the training GPU."""
    if cfg_value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = str(cfg_value).lower().strip()
    if val in ("current", "cuda"):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")
    return torch.device(cfg_value)


def _resolve_grnade_gpu_id(cfg_value):
    """Resolve gRNAde gpu_id, allowing 'current' to follow the training GPU."""
    if cfg_value is None:
        return 0
    val = str(cfg_value).lower().strip()
    if val == "current":
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0
    return int(cfg_value)


class ValidationExternalEvaluator:
    def __init__(self, cfg):
        self._cfg = cfg
        self._grnade = None
        self._rhofold = None
        self._get_features = None
        self._rhofold_device = _resolve_rhofold_device(getattr(cfg, "rhofold_device", None))
        self._grnade_gpu_id = _resolve_grnade_gpu_id(getattr(cfg, "grnade_gpu_id", None))

    def _load_models(self):
        repo_root = getattr(self._cfg, "repo_root", None) or _repo_root()
        from ribocouple.tools.rhofold_api.rhofold.config import rhofold_config
        from ribocouple.tools.rhofold_api.rhofold.rf import RhoFold
        from ribocouple.tools.rhofold_api.rhofold.utils.alphabet import get_features

        seq_source = str(getattr(self._cfg, "sequence_source", "grnade")).lower()
        if seq_source in ("grnade", "both") and self._grnade is None:
            from ribocouple.tools.grnade_api.gRNAde import gRNAde
            with _pushd(repo_root):
                self._grnade = gRNAde(
                    split="all",
                    max_num_conformers=1,
                    gpu_id=self._grnade_gpu_id,
                )

        if self._rhofold is None:
            self._rhofold = RhoFold(rhofold_config)
            ckpt_path = getattr(self._cfg, "rhofold_ckpt", None)
            if not ckpt_path:
                ckpt_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "tools",
                    "rhofold_api",
                    "checkpoints",
                    "RhoFold_pretrained.pt",
                )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    "RhoFold checkpoint not found. Please set "
                    "experiment.validation.external_eval.rhofold_ckpt "
                    f"explicitly. Missing path: {ckpt_path}"
                )
            state = torch.load(
                ckpt_path,
                map_location=torch.device("cpu"),
            )
            self._rhofold.load_state_dict(state["model"])
            self._rhofold.eval()
            self._rhofold = self._rhofold.to(self._rhofold_device)
            self._get_features = get_features

    @torch.no_grad()
    def _predict_rhofold(self, fasta_path: str, output_pdb_path: str):
        feats = self._get_features(fasta_path, fasta_path)
        outputs = self._rhofold(
            tokens=feats["tokens"].to(self._rhofold_device),
            rna_fm_tokens=feats["rna_fm_tokens"].to(self._rhofold_device),
            seq=feats["seq"],
        )
        out = outputs[-1]
        coords = out["cord_tns_pred"][-1].squeeze(0).detach().cpu().numpy()
        conf = out["plddt"][0].detach().cpu().numpy()
        self._rhofold.structure_module.converter.export_pdb_file(
            feats["seq"],
            coords,
            path=output_pdb_path,
            chain_id=None,
            confidence=conf,
            logger=None,
        )

    def _default_summary(self):
        return {
            "ext_n_backbones": 0.0,
            "ext_n_success": 0.0,
            "ext_tm_mean": 0.0,
            "ext_tm_median": 0.0,
            "ext_rmsd_mean": 999.0,
            "ext_rmsd_median": 999.0,
            "ext_tm_valid_ratio": 0.0,
            "ext_rmsd_valid_ratio": 0.0,
            "ext_joint_valid_ratio": 0.0,
        }

    def _select_best(self, records):
        best_by = str(getattr(self._cfg, "best_by", "tm_score"))
        if best_by == "rmsd":
            valid = [r for r in records if r.get("rmsd") is not None]
            if not valid:
                return None
            return min(valid, key=lambda x: x["rmsd"])

        valid = [r for r in records if r.get("tm_score") is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: (x["tm_score"], -(x.get("rmsd") or 1e9)))

    @torch.no_grad()
    def evaluate(self, sample_paths: Iterable[str], save_dir: str):
        self._load_models()

        sample_paths = [os.path.abspath(path) for path in sample_paths]
        if len(sample_paths) == 0:
            return self._default_summary()

        os.makedirs(save_dir, exist_ok=True)
        fasta_dir = os.path.join(save_dir, "fastas")
        pred_dir = os.path.join(save_dir, "preds")
        os.makedirs(fasta_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        seq_source = str(getattr(self._cfg, "sequence_source", "grnade")).lower()
        if seq_source not in ("grnade", "e4", "both"):
            raise ValueError(f"Unknown sequence_source: {seq_source}. Use 'grnade', 'e4', or 'both'.")

        n_samples = int(self._cfg.n_seqs_per_backbone)
        temperature = float(self._cfg.temperature)
        tm_threshold = float(self._cfg.tm_threshold)

        sources_to_run = ["grnade", "e4"] if seq_source == "both" else [seq_source]
        rmsd_threshold = float(self._cfg.rmsd_threshold)
        tmscore_bin = getattr(self._cfg, "tmscore_bin", None)
        if not tmscore_bin:
            import sys as _sys
            _bin_name = "TMscore_macos" if _sys.platform == "darwin" else "TMscore"
            tmscore_bin = os.path.join(
                getattr(self._cfg, "repo_root", None) or _repo_root(),
                "tools",
                _bin_name,
            )
        if not os.path.isfile(tmscore_bin):
            raise FileNotFoundError(
                "TMscore binary not found. Please set "
                "experiment.validation.external_eval.tmscore_bin explicitly. "
                f"Missing path: {tmscore_bin}"
            )

        best_rows = []
        per_sequence_rows = []
        error_rows = []

        for src in sources_to_run:
            for pdb_path in sample_paths:
                pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]

                if src == "grnade":
                    try:
                        seq_records, _, perplexity, recovery = self._grnade.design_from_pdb_file(
                            pdb_filepath=pdb_path,
                            output_filepath=None,
                            n_samples=n_samples,
                            temperature=temperature,
                        )
                    except Exception as exc:
                        error_rows.append(
                            {
                                "generated_pdb": pdb_path,
                                "stage": "grnade",
                                "error": str(exc),
                            }
                        )
                        continue
                    source_sequences = [
                        {
                            "seq": _seqrecord_to_str(seq_record),
                            "seq_idx": seq_idx,
                            "perplexity": float(perplexity[seq_idx]),
                            "recovery": float(recovery[seq_idx]),
                        }
                        for seq_idx, seq_record in enumerate(seq_records)
                    ]
                else:  # src == "e4"
                    fasta_path = pdb_path.replace(".pdb", ".fasta")
                    if not os.path.isfile(fasta_path):
                        error_rows.append(
                            {
                                "generated_pdb": pdb_path,
                                "stage": "e4_fasta",
                                "error": f"Missing sequence-head FASTA: {fasta_path}",
                            }
                        )
                        continue
                    try:
                        seq_record = SeqIO.read(fasta_path, "fasta")
                        seq = _seqrecord_to_str(seq_record)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "generated_pdb": pdb_path,
                                "stage": "e4_fasta",
                                "error": str(exc),
                            }
                        )
                        continue
                    source_sequences = [
                        {
                            "seq": seq,
                            "seq_idx": 0,
                            "perplexity": None,
                            "recovery": None,
                        }
                    ]

                seq_results = []
                for seq_info in source_sequences:
                    seq_idx = seq_info["seq_idx"]
                    seq = seq_info["seq"]
                    src_tag = f"{src}_" if seq_source == "both" else ""
                    fasta_path = os.path.join(fasta_dir, f"{pdb_stem}_{src_tag}seq{seq_idx}.fasta")
                    pred_pdb_path = os.path.join(pred_dir, f"{pdb_stem}_{src_tag}seq{seq_idx}_rhofold.pdb")

                    try:
                        _write_single_seq_fasta(seq, f"{pdb_stem}_{src_tag}seq{seq_idx}", fasta_path)
                        self._predict_rhofold(fasta_path, pred_pdb_path)
                        score_dict = _run_tmscore(tmscore_bin, pdb_path, pred_pdb_path)
                        row = {
                            "generated_pdb": pdb_path,
                            "pdb_name": pdb_stem,
                            "seq_idx": seq_idx,
                            "sequence": seq,
                            "sequence_source": src,
                            "fasta_path": fasta_path,
                            "pred_pdb_path": pred_pdb_path,
                            "perplexity": seq_info["perplexity"],
                            "recovery": seq_info["recovery"],
                            "tm_score": score_dict.get("tm_score"),
                            "rmsd": score_dict.get("rmsd"),
                            "gdt_ts": score_dict.get("gdt_ts"),
                            "gdt_ha": score_dict.get("gdt_ha"),
                        }
                        seq_results.append(row)
                        per_sequence_rows.append(row)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "generated_pdb": pdb_path,
                                "stage": f"seq_{seq_idx}",
                                "error": str(exc),
                                "pred_pdb_path": pred_pdb_path,
                            }
                        )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                best = self._select_best(seq_results)
                if best is None:
                    continue

                best = dict(best)
                best["tm_valid"] = int((best.get("tm_score") or 0.0) >= tm_threshold)
                best["rmsd_valid"] = int(
                    best.get("rmsd") is not None and best["rmsd"] <= rmsd_threshold
                )
                best["joint_valid"] = int(best["tm_valid"] == 1 and best["rmsd_valid"] == 1)
                best_rows.append(best)

        if best_rows:
            summary = {"ext_n_backbones": float(len(sample_paths)), "ext_sequence_source": seq_source}
            for src in sources_to_run:
                src_best_rows = [row for row in best_rows if row.get("sequence_source") == src]
                if not src_best_rows:
                    continue
                # Keep original metric names for single source; add suffix only for 'both'.
                suffix = f"_{src}" if seq_source == "both" else ""
                tm_values = [row["tm_score"] for row in src_best_rows if row.get("tm_score") is not None]
                rmsd_values = [row["rmsd"] for row in src_best_rows if row.get("rmsd") is not None]
                tm_values_sorted = sorted(tm_values)
                rmsd_values_sorted = sorted(rmsd_values)
                summary[f"ext_n_success{suffix}"] = float(len(src_best_rows))
                summary[f"ext_tm_mean{suffix}"] = float(sum(tm_values) / len(tm_values)) if tm_values else 0.0
                summary[f"ext_tm_median{suffix}"] = float(tm_values_sorted[len(tm_values_sorted) // 2]) if tm_values else 0.0
                summary[f"ext_rmsd_mean{suffix}"] = float(sum(rmsd_values) / len(rmsd_values)) if rmsd_values else 999.0
                summary[f"ext_rmsd_median{suffix}"] = float(rmsd_values_sorted[len(rmsd_values_sorted) // 2]) if rmsd_values else 999.0
                summary[f"ext_tm_valid_ratio{suffix}"] = float(sum(row["tm_valid"] for row in src_best_rows) / len(src_best_rows))
                summary[f"ext_rmsd_valid_ratio{suffix}"] = float(sum(row["rmsd_valid"] for row in src_best_rows) / len(src_best_rows))
                summary[f"ext_joint_valid_ratio{suffix}"] = float(sum(row["joint_valid"] for row in src_best_rows) / len(src_best_rows))
        else:
            summary = self._default_summary()
            summary["ext_n_backbones"] = float(len(sample_paths))
            summary["ext_sequence_source"] = seq_source

        best_csv = os.path.join(save_dir, "external_eval_best.csv")
        per_seq_csv = os.path.join(save_dir, "external_eval_per_sequence.csv")
        error_csv = os.path.join(save_dir, "external_eval_errors.csv")
        summary_json = os.path.join(save_dir, "external_eval_summary.json")

        if best_rows:
            with open(best_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(best_rows[0].keys()))
                writer.writeheader()
                writer.writerows(best_rows)

        if per_sequence_rows:
            with open(per_seq_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(per_sequence_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_sequence_rows)

        if error_rows:
            with open(error_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(error_rows[0].keys()))
                writer.writeheader()
                writer.writerows(error_rows)

        with open(summary_json, "w") as handle:
            json.dump(summary, handle, indent=2)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return summary
