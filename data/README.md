# Data directory

- `review_results/`: Precomputed leave-one-cell-line-out predictions (`48` model-fold arrays), paired cell-bootstrap draws, 100-sequence five-context ranking tables, and `verify_results.py`.
- Large sequence-masked count matrices (`AlphaApta_SingleCell_Count_Matrices.zip`), Round-15 abundance records (`AlphaApta_Round15_SELEX_SequenceRedacted.tsv.zst`), and model checkpoints (`AlphaApta_Model_Checkpoints_v2.tar.zst`) are retrieved from [Zenodo (10.5281/zenodo.18242287)](https://doi.org/10.5281/zenodo.18242287) via `bash scripts/download_data.sh`.
