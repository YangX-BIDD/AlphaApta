# Model checkpoints

Download `AlphaApta_Model_Checkpoints_v2.tar.zst` via `bash scripts/download_data.sh` (or directly from [Zenodo 10.5281/zenodo.18242287](https://doi.org/10.5281/zenodo.18242287)) and extract:

```bash
tar --zstd -xf downloads/AlphaApta_Model_Checkpoints_v2.tar.zst -C models/checkpoints/
```

Extracted layout under `models/checkpoints/AlphaApta_model_artifacts_v2/`:

- `D1_D2/`: `d1_encoder.pt`, `hyenadna-small-32k-seqlen-hf/`, and `d2_oracle_gnn_transformer_seed42_fullwnn.pt`.
- `D3_LOCLO/`: Six-fold `cell_encoder/*.pt` and `ebm/*.pt` checkpoints validated by `AlphaAptaD3Runtime`.
