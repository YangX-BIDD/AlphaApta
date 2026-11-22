# Data and analysis scripts for peer review

Run `python verify_results.py` in this directory with Python 3.9 or later, NumPy, pandas, SciPy and scikit-learn.

| Directory | Contents |
| --- | --- |
| `LOCLO/` | Predictions and labels for eight models across six held-out cellular sources (seed-42 benchmark protocol), fold and summary metrics, and paired comparisons. |
| `R3_8_100_sequences/` | Cell-conditioned D2 scores across five profiled cell lines, Round-15 frequencies, ranks, and top-20 lists for the 100-record Round-15 evaluation set (`source_record = 1` corresponds to `APT-1`). |

The script verifies all 48 fold-level ROC-AUC and average precision values, their event-weighted and equal-source means, 42 paired ROC-AUC intervals, and the five-context score-abundance correlations and top-20 overlaps (`7/20` in each cellular context).

For within-fold uncertainty, the supplied ROC-AUC draws represent 1,000 paired cell-cluster bootstrap samples with model predictions fixed.

Event identifiers in each prediction file are fold-local row indices aligned across the eight models, and sequence records are indexed by anonymous row number. Primary single-cell RNA/AbSeq and sequence-masked Stage 2 count matrices (`AlphaApta_SingleCell_Count_Matrices.zip`), sequence-masked Round-15 abundance records (`AlphaApta_Round15_SELEX_SequenceRedacted.tsv.zst`), and trained D1/D2/D3 model checkpoints (`AlphaApta_Model_Checkpoints_v2.tar.zst`) are openly distributed via [Zenodo (10.5281/zenodo.18242287)](https://doi.org/10.5281/zenodo.18242287).
