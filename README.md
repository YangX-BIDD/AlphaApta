# AlphaApta: Cell-Conditioned Aptamer Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18242287.svg)](https://doi.org/10.5281/zenodo.18242287)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

AlphaApta is a proof-of-concept multimodal computational framework for cell-conditioned aptamer design linking an aptamer sequence encoder (D1), a multimodal single-cell aptamer-cell interaction predictor (D2), and a cell-conditioned energy-based sequence generator (D3).

## Architecture

1. **D1 sequence pretraining.** `109,321,575` normalized Round-15 Cell-SELEX sequence-frequency rows (`108,947,560` length-eligible `10-42-nt` records; `98,052,804` in the 90% training partition). Mean-pooled 256-dimensional HyenaDNA token states predict standardized `log10` sequence frequency under mean-squared error.
2. **D2 aptamer-cell interaction modeling.** Receives frozen D1 token states and a `3,257`-feature cellular vector (`50` WNN coordinates derived from RNA and AbSeq, `3,000` RNA features, and `207` AbSeq surface proteins) to predict binary single-cell aptamer detection events across held-out cellular sources.
3. **D3 cell-conditioned generation.** Conditions a bidirectional Transformer energy model on the frozen 256-dimensional D2 cellular representation and samples `40-nt` `A/C/G/T` candidates via 20 Langevin updates.

## Installation

```bash
git clone https://github.com/YangX-BIDD/AlphaApta.git
cd AlphaApta
conda env create -f environment.yml
conda activate alphaapta
pip install -e .
```

## Quick verification

Recompute all 48 leave-one-cell-line-out fold metrics, paired cell-bootstrap intervals, and 100-sequence five-context ranking statistics:

```bash
cd data/review_results && python verify_results.py
```

See [USAGE.md](USAGE.md) for D1, D2, and D3 checkpoint inference examples, or access the interactive online research platform at [alphaapta.ai](https://alphaapta.ai).

## Open data and model checkpoints (Zenodo)

Large data archives and trained model weights are openly distributed via [Zenodo (10.5281/zenodo.18242287)](https://doi.org/10.5281/zenodo.18242287):

- `AlphaApta_Model_Checkpoints_v2.tar.zst`: D1 HyenaDNA encoder, registered D2 GNN-Transformer checkpoint, and six-fold LOCLO D3 cell-encoder and EBM checkpoints.
- `AlphaApta_SingleCell_Count_Matrices.zip`: BD Rhapsody RNA and AbSeq RSEC UMI count matrices (`16,525 x 32,000`), `15,500`-cell WNN50 metadata, and the sequence-masked Stage 2 single-cell aptamer UMI count matrix (`115,529 x 15,500`).
- `AlphaApta_Round15_SELEX_SequenceRedacted.tsv.zst`: All `109,321,575` Round-15 Cell-SELEX relative-frequency records with nucleotide strings masked by length-matched `N` characters (`R15_000000001` to `R15_109321575`).

Download directly with:

```bash
bash scripts/download_data.sh
```

## Repository layout

```text
AlphaApta/
|-- alphaapta/            D1, D2, and D3 model and runtime modules
|-- configs/              Audited D1, D2, and D3 YAML configurations
|-- data/review_results/  LOCLO predictions, bootstrap draws, and verification script
|-- models/               Local checkpoint placement guide
|-- scripts/              Inference and Zenodo download utilities
|-- tests/                Model and checkpoint compatibility tests
|-- USAGE.md              Inference usage examples
`-- PAPER_STATEMENTS.md   Data and code availability statement
```

## License

Source code is released under the [MIT License](LICENSE). Deposited data and model checkpoints on Zenodo are released under CC-BY-4.0.
