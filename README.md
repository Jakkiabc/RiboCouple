# RiboCouple

**RiboCouple links RNA backbone generation to sequence design through a learned interaction topology.**

RiboCouple infers a probabilistic residue-pair state from generated RNA backbone geometry, recycles binary pair probabilities through a second structural pass, and transfers binary and typed pair signals to sequence decoding as soft attention biases.

**Preprint:** https://doi.org/10.64898/2026.10.15.759394
**Model weights:** [Google Drive folder](https://drive.google.com/drive/folders/1pD0lKtfeUsxtVxuPf7b-ayFK7E56wLeq?usp=drive_link)

## Usage

```bash
pip install -e .

# Train
export RIBOCOUPLE_TRAIN_CSV=/path/to/training_metadata.csv
export RIBOCOUPLE_BP_LABELS=/path/to/pair_labels
python scripts/train_se3_flows.py
```

### Unconditional generation pipeline (sample → rank → refold)

One command for the whole workflow — sample, score the full pool with the
pair head, then refold only the top-K per length (with an optional random-K
baseline to quantify the reranking gain):

```bash
python scripts/run_generation_pipeline.py \
  --ckpt model.ckpt \
  --out_dir pipeline_out/ \
  --lengths 50 60 70 80 90 100 110 120 130 140 150 \
  --samples-per-length 50 \
  --top-k 5 --random-k \
  --rhofold_ckpt /path/to/RhoFold_pretrained.pt
```

Use `--refold-all` instead of `--top-k` to skip reranking and refold every
sampled backbone (full-pool evaluation):

```bash
python scripts/run_generation_pipeline.py \
  --ckpt model.ckpt \
  --out_dir pipeline_full/ \
  --samples-per-length 50 \
  --refold-all \
  --rhofold_ckpt /path/to/RhoFold_pretrained.pt
```

Or step by step:

```bash
# Sample backbones; checkpoints with a sequence head also write a FASTA per sample
python scripts/sample_backbones.py \
  --ckpt model.ckpt \
  --out-dir generated_backbones \
  --lengths 50 60 70 80 90 100 110 120 130 140 150 \
  --samples-per-length 5 \
  --steps 50 \
  --device cuda

# Rank candidates by pair-head confidence (--select-top N writes a top-N shortlist)
python scripts/score_pair_confidence.py \
  --ckpt model.ckpt \
  --pdb_dir backbones/ \
  --out pair_scores.csv

# Refold the sampled sequences with RhoFold and compare to the sampled backbones
# (reads the <name>.fasta next to each <name>.pdb; use --fasta_dir if your
#  sequences live elsewhere, or --pred_dir to score pre-folded PDBs instead.
#  RhoFold code is bundled, weights are not — get RhoFold_pretrained.pt from
#  the upstream RhoFold release)
python scripts/evaluate_refold.py \
  --backbone_dir generated_backbones/ \
  --rhofold_ckpt /path/to/RhoFold_pretrained.pt \
  --out_dir refold_eval/
```

Evaluation here is refolding validity (TM-score / RMSD against the sampled
backbone; joint validity at TM ≥ 0.45 and RMSD ≤ 5.5, as in the manuscript).
An optional external control is to design sequences for the same backbone pool
with gRNAde (not bundled) and run `evaluate_refold.py --pred_dir` on its
refolded outputs.

### Fixed-backbone sequence design (inverse folding)

```bash
# Decode a sequence for a supplied backbone
python scripts/decode_pair_coupled.py \
  --ckpt model.ckpt \
  --pdb backbone.pdb \
  --rule greedy \
  --out designed_sequence.fasta
```

`--rule greedy` is the pair-coupled decoder from the manuscript;
`--rule argmax` is plain argmax decoding (higher native-sequence recovery,
lower pair agreement); `--rule mutual` is the mutual-best variant.

```bash
# Evaluate fixed-backbone designs against the native sequence/structure:
# recovery + 2D MCC + BP-F1 (no refolding needed)
python scripts/evaluate_inverse_folding.py \
  --pdb_dir native_pdbs/ \
  --fasta_dir designed_fastas/ \
  --out_dir if_eval/
```

The evaluator reports two ground-truth tracks, as in the manuscript: a
canonical track (x3dna `find_pair` by default, or `--pairs_engine dssr`)
and a structural H-bond track computed in pure Python. Native PDBs should
be cleaned, single-chain RNA structures, as in the benchmark. If neither
x3dna nor DSSR is available, `--pairs_engine none` reports recovery and
BP-F1_hbond only (ViennaRNA is still required), and `--recovery-only`
skips all folding-based metrics.

## License

Apache-2.0 (see [LICENSE](LICENSE)); third-party attributions in [NOTICE.md](NOTICE.md).

## Contact

Zhiqi Ma — zhiqima@link.cuhk.edu.cn
