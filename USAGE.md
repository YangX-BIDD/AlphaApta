# AlphaApta usage

## Install

```bash
conda env create -f environment.yml
conda activate alphaapta
pip install -e .
```

## D3 cell-conditioned generation

The final D3 model accepts a 256-dimensional cellular condition produced by the
frozen D2 cell-context encoder. It returns complete 40-nt A/C/G/T sequences.

```python
import torch

from alphaapta.models import (
    CellConditionedEnergyGenerator,
    D3EnergyConfig,
    decode_tokens,
)

config = D3EnergyConfig()
model = CellConditionedEnergyGenerator(config)
payload = torch.load("models/checkpoints/AlphaApta_model_artifacts_v2/D3_LOCLO/ebm/AGS.pt", map_location="cpu")
model.load_state_dict(payload.get("state_dict", payload), strict=True)
model.eval()

condition = torch.load("example_cell_condition.pt").float().reshape(1, 256)
tokens = model.generate(condition, temperature=1.0)
sequences = decode_tokens(tokens)
print(sequences)
```

The default sampler initializes zero nucleotide logits and applies 20 Langevin
updates with gradient step size 10.0 and Gaussian noise standard deviation
0.005. PAD is excluded before sampling. Top-p and top-k truncation are not used.

## Cellular input layout

The D2 cell-context encoder uses a 3,257-value vector in this fixed order:

1. 50 WNN coordinates derived from RNA and AbSeq;
2. 3,000 RNA values;
3. 207 AbSeq values.

Aptamer barcode counts are not part of this vector. They are used only to
construct D2 binary supervision labels.

## Reproducing model definitions

The audited settings are recorded in:

- `configs/pretrain_d1_regression_config.yaml`
- `configs/d2_config.yaml`
- `configs/d3_energy_based_config.yaml`

The final D3 architecture is implemented in
`alphaapta/models/d3_energy_based_generator.py`.

## D1 sequence-frequency prediction

The deposited D1 checkpoint contains the HyenaDNA encoder, scalar regression
head, target mean and standard deviation, and retained epoch. For example:

```bash
python scripts/predict_d1_frequency.py \
  --checkpoint models/checkpoints/AlphaApta_model_artifacts_v2/D1_D2/d1_encoder.pt \
  --model-directory models/checkpoints/AlphaApta_model_artifacts_v2/D1_D2/hyenadna-small-32k-seqlen-hf \
  --sequence ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT \
  --output d1_prediction.json
```
