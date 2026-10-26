# Third-party notices (release)

## Upstream attribution
- **RNA-FrameFlow** (Anand et al., TMLR 2025; github.com/rish-16/rna-backbone-design):
  RiboCouple is developed on top of the RNA-FrameFlow codebase. New modules added on top of
  the upstream code include: pair-recycled trunk, binary/typed pair heads, pair-biased
  Transformer sequence decoder, the inference-time intervention suite (donor substitution /
  rewiring / interpolation / masking), and pair-confidence candidate prioritization.

- **RhoFold** (structure evaluator): the RhoFold inference code is vendored at
  `src/ribocouple/tools/rhofold_api/` (Apache-2.0; original license headers retained in
  source files). It embeds a vendored copy of the RNA-FM model code derived from
  fair-esm (Facebook/Meta, MIT license; headers retained). The pretrained weights are
  NOT included; download `RhoFold_pretrained.pt` from the upstream project
  (https://github.com/CUHKAIRobotics/rhofold) and pass it via `--rhofold_ckpt`.
- **TMscore** (structure metric): `tools/TMscore` and `tools/TMscore_macos` are
  redistributed with the upstream `TMscore.cpp` source. The source header grants
  permission to use, copy, modify, and distribute with its notices and reference
  information preserved; see https://zhanggroup.org/TM-score/TMscore.cpp.
  The bundled PStreams component carries the Boost Software License 1.0;
  its license text is at `tools/LICENSE_1_0.txt`. Cite Zhang and Skolnick,
  Proteins 57:702–710 (2004).
- **x3dna / DSSR** (pair annotation): third-party; used at evaluation time only for the
  canonical BP-F1 / 2D MCC track (`--pairs_engine find_pair`, paper default, or
  `--pairs_engine dssr`). Neither is bundled. X3DNA v2.4 is distributed by its authors
  under CC-BY-NC-4.0 (non-commercial, attribution required); DSSR basic is under its own
  license that forbids redistribution — obtain both from https://x3dna.org. Users without
  either binary can run `--pairs_engine none` for the license-free H-bond BP-F1 track
  (pure Python) or `--recovery-only` for recovery alone.
- **ViennaRNA** (RNA.fold): GPL-licensed; used at evaluation time for pairing; not bundled.
- **OpenBabel**, **DGL**, **PyTorch**, **e3nn**, **ProDy**, **Biopython**: dependencies per
  `environment.yml`; their own licenses apply.
- **gRNAde / NA-MPNN / RNA-FrameFlow / RiboDiffusion / RDesign / RIDER**: external methods used
  only in benchmark comparisons (not part of this package; see manuscript supplement).

## Data
- Example / demo structures (PDB ids) belong to their original depositors.

## License of this package
Apache-2.0 (see LICENSE).


## Additional inherited components
- Portions of the geometric flow, model, and utility code retain upstream-derived components or adaptations from Microsoft's Protein Frame Flow (https://github.com/microsoft/protein-frame-flow); see source-file notices.
- Portions of RNA data parsing, atom representations, and data utilities retain curated/adapted components from MMDiff (https://github.com/Profluent-Internships/MMDiff); see source-file notices.
